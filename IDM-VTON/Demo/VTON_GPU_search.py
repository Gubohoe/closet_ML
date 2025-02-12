#피팅 + 유사도 검색
import sys
sys.path.append('./')
import uvicorn, pathlib
from PIL import Image
import torch
import google.generativeai as genai
from typing import List, Tuple
import os
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from fastapi import FastAPI, File, UploadFile, Response, Form
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import shutil
import io

# Import models
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
   CLIPImageProcessor,
   CLIPVisionModelWithProjection,
   CLIPTextModel,
   CLIPTextModelWithProjection,
   AutoTokenizer,
)
from diffusers import DDPMScheduler, AutoencoderKL

# FastAPI 앱 생성
app = FastAPI()

# Gemini API 설정
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)
persona = "Please write a description of this image in 1 or 2 lines ex:a plain dusty blue cotton sweatshirt with crew neck and long sleeves"
model = genai.GenerativeModel('gemini-2.0-flash-001', system_instruction=persona)

# 전역 변수
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
collection_name = "garment_vectors"
model_path = 'yisol/IDM-VTON'

# 모델들을 전역 변수로 선언
pipe = None
parsing_model = None
openpose_model = None
tensor_transform = None
feature_transform = None
qdrant_client = None


#사용 모델 초기화
def initialize_models(base_path: str) -> Tuple:
   unet = UNet2DConditionModel.from_pretrained(
       base_path, subfolder="unet", torch_dtype=torch.float16
   ).requires_grad_(False)
   
   tokenizer_one = AutoTokenizer.from_pretrained(
       base_path, subfolder="tokenizer", use_fast=False
   )
   tokenizer_two = AutoTokenizer.from_pretrained(
       base_path, subfolder="tokenizer_2", use_fast=False
   )
   
   text_encoder_one = CLIPTextModel.from_pretrained(
       base_path, subfolder="text_encoder", torch_dtype=torch.float16
   ).requires_grad_(False)
   
   text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
       base_path, subfolder="text_encoder_2", torch_dtype=torch.float16
   ).requires_grad_(False)
   
   image_encoder = CLIPVisionModelWithProjection.from_pretrained(
       base_path, subfolder="image_encoder", torch_dtype=torch.float16
   ).requires_grad_(False)
   
   vae = AutoencoderKL.from_pretrained(
       base_path, subfolder="vae", torch_dtype=torch.float16
   ).requires_grad_(False)
   
   noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
   
   unet_encoder = UNet2DConditionModel_ref.from_pretrained(
       base_path, subfolder="unet_encoder", torch_dtype=torch.float16
   ).requires_grad_(False)
   
   pipe = TryonPipeline.from_pretrained(
       base_path,
       unet=unet,
       vae=vae,
       feature_extractor=CLIPImageProcessor(),
       text_encoder=text_encoder_one,
       text_encoder_2=text_encoder_two,
       tokenizer=tokenizer_one,
       tokenizer_2=tokenizer_two,
       scheduler=noise_scheduler,
       image_encoder=image_encoder,
       torch_dtype=torch.float16,
   )
   pipe.unet_encoder = unet_encoder

   parsing_model = Parsing(0)
   openpose_model = OpenPose(0)

   # 피팅용 transform
   tensor_transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize([0.5], [0.5]),
   ])

   # 유사도 검색용 transform
   feature_transform = transforms.Compose([
       transforms.Resize((768, 1024)),
       transforms.ToTensor(),
       transforms.Normalize([0.5], [0.5]),
   ])

   return pipe, parsing_model, openpose_model, tensor_transform, feature_transform

#이미지 크롭 및 리사이즈
def crop_and_resize_image(image: Image.Image) -> Tuple[Image.Image, tuple]:
    width, height = image.size
    target_width = int(min(width, height * (3 / 4)))       #이미지 3:4비율 계산
    target_height = int(min(height, width * (4 / 3)))       #이미지 4:3비율 계산
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = (width + target_width) // 2
    bottom = (height + target_height) // 2          #이미지 중앙 기준
    crop_coords = (left, top, right, bottom)        #크롭 진행
    return image.crop(crop_coords).resize((768, 1024)), crop_coords # 크롭 이미지 리사이즈

#마스크 생성
def generate_mask(parsing_model, openpose_model, image: Image.Image, cloth_type: str) -> Image.Image:
    keypoints = openpose_model(image.resize((384, 512)))        #openpose 모델로 부터 사람 키포인트 데이터 추출
    model_parse, _ = parsing_model(image.resize((384, 512)))        #humanParsing 모델로부터 사람 파싱 데이터 추출
    mask, _ = get_mask_location('hd', cloth_type, model_parse, keypoints)   #두 데이터를 입력으로 마스크 생성
    return mask.resize((768, 1024))

def prepare_pose_image(image: Image.Image) -> Image.Image:
    
    # DensePose가 추출한 IUV 정보(3D 표면 매핑)를 시각화된 2D 이미지로 렌더링합니다
    # 각 신체 부위는 서로 다른 색상으로 표현
    # UV 좌표값은 색상의 그라데이션으로 표현
    
   image_arg = _apply_exif_orientation(image.resize((384, 512)))
   image_arg = convert_PIL_to_numpy(image_arg, format="BGR")
   args = apply_net.create_argument_parser().parse_args((
       'show',
       './configs/densepose_rcnn_R_50_FPN_s1x.yaml',
       './ckpt/densepose/model_final_162be9.pkl',
       'dp_segm',
       '-v',
       '--opts',
       'MODEL.DEVICE',
       'cuda'
   ))
   pose_img = args.func(args, image_arg)
   pose_img = pose_img[:,:,::-1]
   return Image.fromarray(pose_img).resize((768, 1024))

#Virtual Try-On 진행
def generate_tryon_image(pipe, human_image: Image.Image, garment_image: Image.Image,
                       mask: Image.Image, garment_description: str,
                       denoise_steps: int = 50, seed: int = None) -> Image.Image:
   
   pose_img = prepare_pose_image(human_image)   #DensePose 모델을 통해 인체 포즈 이미지 생성
   pose_tensor = tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)  #포즈 이미지 텐서로 변환
   garm_tensor = tensor_transform(garment_image).unsqueeze(0).to(device, torch.float16)  #옷 이미지 텐서로 변환

   pipe.to(device)  #pipe GPU 이동

   generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

   prompt = f"model is wearing {garment_description}"
   negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    #프롬프트 임베딩 생성(CLIP)
   with torch.inference_mode():  #추론모드에서 텍스트 프롬프트 벡터 임베딩
       #전체 가상 피팅에 이미지 생성을 위한 임베딩
       prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
           pipe.encode_prompt(
               prompt,
               num_images_per_prompt=1,
               do_classifier_free_guidance=True,
               negative_prompt=negative_prompt,
           )
       )

        #의상의 특징을 위한 임베딩
       prompt = f"a photo of {garment_description}"     #의상에 대한 프롬프트 벡터 임베딩
       prompt_embeds_c, _, _, _ = pipe.encode_prompt(
           [prompt],
           num_images_per_prompt=1,
           do_classifier_free_guidance=False,
           negative_prompt=[negative_prompt],
       )

    #생성한 마스크, 벡터값, 이미지 등 float16으로 변환 및 GPU로 이동, 피팅진행
   images = pipe(
       prompt_embeds=prompt_embeds.to(device, torch.float16),
       negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
       pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
       negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
       num_inference_steps=denoise_steps,
       generator=generator,
       strength=1.0,
       pose_img=pose_tensor,
       text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
       cloth=garm_tensor,
       mask_image=mask,
       image=human_image,
       height=1024,
       width=768,
       ip_adapter_image=garment_image.resize((768, 1024)),
       guidance_scale=2.0,
   )[0]

   return images[0]

def extract_features(image: Image.Image, target_dims: int = 65536):
    image_tensor = feature_transform(image)  #옷 이미지 텐서로 변환
    mask = torch.zeros((1, image_tensor.shape[1], image_tensor.shape[2]))    #이미지와 같은 크기의 0으로 채워진 마스크 생성
    image_tensor = torch.cat([image_tensor, mask], dim=0)    #이미지 텐서와 마스크 텐서 결합
    image_tensor = image_tensor.unsqueeze(0).to(device, torch.float16)   #배치 차원을 추가하고 float16으로 변환 및 GPU로 이동

    #Unet_ref에 필요한 시간 스템프, 안코더 상태 초기화
    timesteps = torch.zeros(1, device=device) 
    encoder_hidden_states = torch.zeros(1, 77, 2048, device=device, dtype=torch.float16)

    with torch.no_grad():   #그레디언트 비활성화(추론만)
        #Unet_ref를 통해 이미지 특징 추출
        _, garment_features = pipe.unet_encoder(
            image_tensor,
            timesteps,
            encoder_hidden_states,
            return_dict=False
        )

        #특징 벡터 저장 리스트
        feature_vectors = []
        
        # 각 특징 맵의 차원에 따라 적절한 평균값 계산
        for feat in garment_features:
            if len(feat.shape) == 2:     # 2D 텐서는 그대로 사용
                feat_mean = feat
            elif len(feat.shape) == 3:    # 3D 텐서는 마지막 차원으로 평균
                feat_mean = feat.mean(dim=2)
            elif len(feat.shape) == 4:    # 4D 텐서는 마지막 두 차원으로 평균
                feat_mean = feat.mean(dim=[2, 3])
            else:
                continue
            feature_vectors.append(feat_mean.cpu())

        # 모든 특징 벡터를 하나로 연결
        all_features = torch.cat(feature_vectors, dim=1)
        
        # 목표 차원 크기에 맞게 특징 맵 크기 조정
        target_latent_size = int(np.sqrt(target_dims // 4))
        input_side_length = int(target_latent_size / 0.13025)
        needed_features = 3 * input_side_length * input_side_length
        
        # RGB 이미지 형태로 특징 재구성
        features_rgb = all_features[:, :needed_features].reshape(1, 3, input_side_length, input_side_length)
        features_rgb = features_rgb.to(device, torch.float16)
        
        # -1에서 1 사이로 정규화
        features_rgb = 2 * (features_rgb - features_rgb.min()) / (features_rgb.max() - features_rgb.min()) - 1
        
        # VAE로 잠재 공간으로 인코딩
        latents = pipe.vae.encode(features_rgb).latent_dist.sample()
        features = latents.flatten().cpu().numpy()
        
        # 목표 차원 수에 맞게 자르기
        if len(features) > target_dims:
            features = features[:target_dims]
        
        return features

def search_similar_items(client: QdrantClient, vector: np.ndarray, category: str, top_k: int = 3, score_threshold: float = 0.85):
    # 카테고리 필터 설정
    search_filter = Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value=category)
            )
        ]
    )
    
    # 초기 검색 limit 설정 (top_k의 3배)
    initial_limit = top_k * 3
    
    # Qdrant에서 유사한 벡터 검색
    results = client.search(
        collection_name=collection_name,
        query_vector=vector.tolist(),
        limit=initial_limit,
        query_filter=search_filter,
        score_threshold=score_threshold
    )
    
    # 임계값 이상의 결과만 필터링
    filtered_results = [
        result for result in results
        if result.score >= score_threshold
    ]
    
    # 점수 기준으로 내림차순 정렬
    filtered_results.sort(key=lambda x: x.score, reverse=True)
    
    # 상위 k개 결과 선택
    top_results = filtered_results[:top_k]

    # 결과 이미지 URL들 반환
    return [result.payload["image_url"] for result in top_results]

#서버 실행할 때 모델들 초기화
@app.on_event("startup")
async def startup_event():
   global pipe, parsing_model, openpose_model, tensor_transform, feature_transform, qdrant_client
   pipe, parsing_model, openpose_model, tensor_transform, feature_transform = initialize_models(model_path)
   qdrant_client = QdrantClient(host="localhost", port=6333)
   
@app.on_event("shutdown")
async def shutdown_event():
    global pipe, parsing_model, openpose_model
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 모델 메모리 해제
    pipe = None
    parsing_model = None
    openpose_model = None

@app.post("/try-on")
async def try_on(
   human_image: UploadFile = File(...),
   garment_image: UploadFile = File(...),
   cloth_type: str = Form(...),
   denoise_steps: int = 40,
   seed: int = 42
):
   try:
       print(f"Received cloth_type: {cloth_type}")
       human_img = Image.open(io.BytesIO(await human_image.read())).convert("RGB")
       garment_img = Image.open(io.BytesIO(await garment_image.read())).convert("RGB").resize((768, 1024))

       garment_bytes = io.BytesIO()
       garment_img.save(garment_bytes, format='PNG')
       garment_bytes = garment_bytes.getvalue()

       image = {
           'mime_type': 'image/png',
           'data': garment_bytes
       }
       response = model.generate_content([persona, image])
       garment_description = response.text
       print(garment_description)

       human_img_processed, _ = crop_and_resize_image(human_img)
       mask = generate_mask(parsing_model, openpose_model, human_img_processed, cloth_type)

       result = generate_tryon_image(
           pipe,
           human_img_processed,
           garment_img,
           mask,
           garment_description,
           denoise_steps,
           seed
       )

       buffered = io.BytesIO()
       result.save(buffered, format="PNG")

       return Response(content=buffered.getvalue(), media_type="image/png")

   except Exception as e:
       print(f"Error: {e}")
       raise e

@app.post("/try-on-full-outfit")
async def try_on_full_outfit(
   human_image: UploadFile = File(...),
   top_image: UploadFile = File(...),
   bottom_image: UploadFile = File(...),
):
   try:
       denoise_steps = 40
       seed = 42

       human_img = Image.open(io.BytesIO(await human_image.read())).convert("RGB")
       top_img = Image.open(io.BytesIO(await top_image.read())).convert("RGB").resize((768, 1024))
       bottom_img = Image.open(io.BytesIO(await bottom_image.read())).convert("RGB").resize((768, 1024))

       # Gemini API로 의상 설명 생성
       top_bytes = io.BytesIO()
       top_img.save(top_bytes, format='PNG')
       top_bytes = top_bytes.getvalue()

       top_image_data = {
           'mime_type': 'image/png',
           'data': top_bytes
       }
       top_response = model.generate_content([persona, top_image_data])
       top_description = top_response.text
       print("Top description:", top_description)

       bottom_bytes = io.BytesIO()
       bottom_img.save(bottom_bytes, format='PNG')
       bottom_bytes = bottom_bytes.getvalue()
       bottom_image_data = {
           'mime_type': 'image/png',
           'data': bottom_bytes
       }
       bottom_response = model.generate_content([persona, bottom_image_data])
       bottom_description = bottom_response.text
       print("Bottom description:", bottom_description)

       # 상의 피팅
       print("Processing top garment...")
       human_img_processed, _ = crop_and_resize_image(human_img)
       top_mask = generate_mask(parsing_model, openpose_model, human_img_processed, "upper_body")

       top_result = generate_tryon_image(
           pipe,
           human_img_processed,
           top_img,
           top_mask,
           top_description,
           denoise_steps,
           seed
       )

       # 중간 결과를 메모리에 저장
       top_result_bytes = io.BytesIO()
       top_result.save(top_result_bytes, format='PNG')
       top_result_img = Image.open(top_result_bytes)

       # 하의 피팅
       print("Processing bottom garment...")
       bottom_mask = generate_mask(parsing_model, openpose_model, top_result_img, "lower_body")

       final_result = generate_tryon_image(
           pipe,
           top_result_img,
           bottom_img,
           bottom_mask,
           bottom_description,
           denoise_steps,
           seed
       )

       buffered = io.BytesIO()
       final_result.save(buffered, format="PNG")

       return Response(content=buffered.getvalue(), media_type="image/png")

   except Exception as e:
       print(f"Error: {e}")
       raise e

@app.post("/search-similar")
async def search_similar(
   garment_image: UploadFile = File(...),
   cloth_type: str = Form(...)
):
   try:
       #이미지 PIL Image+RGB로 변환
       image = Image.open(io.BytesIO(await garment_image.read())).convert("RGB")
       features = extract_features(image)       #특징 추출
       similar_urls = search_similar_items(     #유사도 검색 진행
           qdrant_client,
           features,
           cloth_type
       )
       return {"similar_items": similar_urls}   #해당하는 이미지 URL 반환

   except Exception as e:
       return {"error": str(e)}

@app.post("/search-similar-full")
async def search_similar_full(
   top_image: UploadFile = File(...),
   bottom_image: UploadFile = File(...)
):
   try:
       #상의 하의 따로 진행
       top_img = Image.open(io.BytesIO(await top_image.read())).convert("RGB")
       features_top = extract_features(top_img)

       bottom_img = Image.open(io.BytesIO(await bottom_image.read())).convert("RGB")
       features_bottom = extract_features(bottom_img)

       similar_top = search_similar_items(
           qdrant_client,
           features_top,
           'upper_body'
       )

       similar_bottom = search_similar_items(
           qdrant_client,
           features_bottom,
           'lower_body'
       )

       similar_urls = similar_top + similar_bottom

       return {"similar_items": similar_urls}

   except Exception as e:
       return {"error": str(e)}