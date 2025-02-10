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
import shutil
import io

# Import your custom pipeline and models
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
tensor_transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize([0.5], [0.5]),
])

# GPU 모델들을 전역 변수로 선언
pipe = None
parsing_model = None
openpose_model = None

@app.on_event("startup")
async def startup_event():
   global pipe, parsing_model, openpose_model
   pipe, parsing_model, openpose_model = initialize_models('yisol/IDM-VTON')

@app.on_event("shutdown")
async def shutdown_event():
    global pipe, parsing_model, openpose_model
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 모델 메모리 해제
    pipe = None
    parsing_model = None
    openpose_model = None

# 필요한 함수들
def pil_to_binary_mask(pil_image: Image.Image, threshold: int = 0) -> Image.Image:
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    mask[binary_mask] = 1
    mask = (mask * 255).astype(np.uint8)
    return Image.fromarray(mask)

def crop_and_resize_image(image: Image.Image) -> Tuple[Image.Image, tuple]:
    width, height = image.size
    target_width = int(min(width, height * (3 / 4)))
    target_height = int(min(height, width * (4 / 3)))
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = (width + target_width) // 2
    bottom = (height + target_height) // 2
    crop_coords = (left, top, right, bottom)
    return image.crop(crop_coords).resize((768, 1024)), crop_coords

def generate_mask(parsing_model, openpose_model, image: Image.Image, cloth_type: str) -> Image.Image:
    keypoints = openpose_model(image.resize((384, 512)))
    model_parse, _ = parsing_model(image.resize((384, 512)))
    mask, _ = get_mask_location('hd', cloth_type, model_parse, keypoints)
    return mask.resize((768, 1024))

def prepare_pose_image(image: Image.Image) -> Image.Image:
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

def initialize_models(base_path: str) -> Tuple:
    # Initialize UNet
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

    return pipe, parsing_model, openpose_model

def generate_tryon_image(pipe, human_image: Image.Image, garment_image: Image.Image,
                        mask: Image.Image, garment_description: str,
                        denoise_steps: int = 50, seed: int = None) -> Image.Image:
    
    pose_img = prepare_pose_image(human_image)
    pose_tensor = tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)
    garm_tensor = tensor_transform(garment_image).unsqueeze(0).to(device, torch.float16)
    
    torch.cuda.empty_cache()
    
    pipe.to(device)
    pipe.unet_encoder.to(device)

    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

    prompt = f"model is wearing {garment_description}"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    with torch.inference_mode():
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
            pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
        )

        prompt = f"a photo of {garment_description}"
        prompt_embeds_c, _, _, _ = pipe.encode_prompt(
            [prompt],
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=[negative_prompt],
        )

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

# FastAPI 엔드포인트
# 상의/하의/한벌옷
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

       # Gemini API로 의상 설명 생성
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

       # 이미지 처리
       human_img_processed, _ = crop_and_resize_image(human_img)
       mask = generate_mask(parsing_model, openpose_model, human_img_processed, cloth_type)
       torch.cuda.empty_cache()

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
   finally:
       torch.cuda.empty_cache()

@app.post("/try-on-full-outfit")
async def try_on_full_outfit(
   human_image: UploadFile = File(...),
   top_image: UploadFile = File(...),
   bottom_image: UploadFile = File(...),
):
   try:
       denoise_steps = 40
       seed = 42
       
       # 이미지 로드
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
       torch.cuda.empty_cache()
       
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
       
       torch.cuda.empty_cache()

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
   finally:
       torch.cuda.empty_cache()