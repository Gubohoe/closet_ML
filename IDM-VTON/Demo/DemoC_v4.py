#V4 gemini연결해서 프롬프트 자동생성
import sys
import pathlib
sys.path.append('./')
from PIL import Image, ImageDraw
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

GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)

persona = "Please write a description of this image in 1 or 2 lines ex:a plain dusty blue cotton sweatshirt with crew neck and long sleeves"

model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=persona)

# Global variables
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

torch.cuda.empty_cache()

garment_path = "img/garment/pants2.jpg"

def garment_description_gemini():
    
    image = {
        'mime_type': 'image/png',
        'data': pathlib.Path(garment_path).read_bytes()
    }
    
    response = model.generate_content([persona, image])
    print(f"Description: {response.text}")
    
    return response.text


#모델 로드
def initialize_models(base_path: str) -> Tuple:
    """Initialize all required models and return them"""
    # Initialize UNet
    unet = UNet2DConditionModel.from_pretrained(
        base_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    ).requires_grad_(False)

    # Initialize tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer",
        use_fast=False,
    )

    tokenizer_two = AutoTokenizer.from_pretrained(
        base_path,
        subfolder="tokenizer_2",
        use_fast=False,
    )

    # Initialize other models
    text_encoder_one = CLIPTextModel.from_pretrained(
        base_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    ).requires_grad_(False)

    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    ).requires_grad_(False)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base_path,
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    ).requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(
        base_path,
        subfolder="vae",
        torch_dtype=torch.float16,
    ).requires_grad_(False)

    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        base_path,
        subfolder="unet_encoder",
        torch_dtype=torch.float16,
    ).requires_grad_(False)

    # Initialize pipeline
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

    # Initialize parsing and pose models
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)

    return pipe, parsing_model, openpose_model

#이미지 마스킹(이미지 Numpy배열로 변환 -> 임계값을 기준으로 이진 마스크 생성)
def pil_to_binary_mask(pil_image: Image.Image, threshold: int = 0) -> Image.Image:
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    mask[binary_mask] = 1
    mask = (mask * 255).astype(np.uint8)
    return Image.fromarray(mask)

# 크롭 진행여부(T의 경우 가로 세로 비율에 맞춰 크롭 후 768X1024 크기로 리사이즈, F의 경우 원본 이미지 리사이즈)
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

# 마스크 자동 생성 여부(T의 경우 OpenPose 모델, Parsing 모델을 활용하여 사람의 상체 마스크를 자동 생성 F의 경우 마스크 이미지를 받아 이진 데이터로 변환)
def generate_mask(parsing_model, openpose_model, image: Image.Image, cloth_type: str) -> Image.Image:
    keypoints = openpose_model(image.resize((384, 512)))
    model_parse, _ = parsing_model(image.resize((384, 512)))
    mask, _ = get_mask_location('hd', cloth_type, model_parse, keypoints)
        
    return mask.resize((768, 1024))

# +시각화 코드
# def generate_mask(parsing_model, openpose_model, image: Image.Image, cloth_type: str) -> Image.Image:
#     # OpenPose 실행 및 결과 시각화
#     keypoints = openpose_model(image.resize((384, 512)))
    
#     # OpenPose 결과 시각화
#     openpose_vis = Image.new('RGB', (384, 512), (0, 0, 0))
#     draw = ImageDraw.Draw(openpose_vis)
    
#     # 키포인트 그리기
#     keypoints_data = keypoints['pose_keypoints_2d']
#     for point in keypoints_data:
#         x, y = point[0], point[1]
#         if x > 0 and y > 0:  # 유효한 키포인트만 그리기
#             draw.ellipse([x-3, y-3, x+3, y+3], fill='white')
    
#     # 결과 저장
#     openpose_vis = openpose_vis.resize((768, 1024))
#     openpose_vis.save("result/openpose_output.png")
#     print("OpenPose 결과 이미지가 openpose_output.png로 저장되었습니다.")

#     # Human Parsing 실행
#     model_parse, vis_parsing = parsing_model(image.resize((384, 512)))
    
#     # PIL Image를 numpy 배열로 변환
#     model_parse_np = np.array(model_parse)
#     print("Model parse shape:", model_parse_np.shape)
#     print("Model parse value range:", model_parse_np.min(), model_parse_np.max())

#     # vis_parsing 처리
#     if vis_parsing is not None:
#         if isinstance(vis_parsing, Image.Image):
#             vis_parsing_np = np.array(vis_parsing)
#         else:
#             vis_parsing_np = vis_parsing
        
#         # 시각화를 위한 컬러맵 적용
#         colored_parse = np.zeros((model_parse_np.shape[0], model_parse_np.shape[1], 3), dtype=np.uint8)
#         for label in range(int(model_parse_np.max()) + 1):
#             mask = model_parse_np == label
#             color = np.random.randint(0, 255, 3)
#             colored_parse[mask] = color
            
#         # 결과 저장
#         Image.fromarray(colored_parse).resize((768, 1024)).save("result/humanparsing_output.png")
#         print("Human Parsing 결과 이미지가 humanparsing_output.png로 저장되었습니다.")

#     # 마스크 생성
#     mask, _ = get_mask_location('hd', cloth_type, model_parse, keypoints)
#     mask_image = mask.resize((768, 1024))
    
#     # 마스크 시각화를 위한 mask_gray 생성
#     mask_gray = (1 - transforms.ToTensor()(mask_image)).unsqueeze(0) * tensor_transform(image)
#     mask_gray = to_pil_image((mask_gray.squeeze().clamp(-1, 1) + 1) / 2)
     
#     try:
#         mask_image.save("result/mask_output.png")
#         mask_gray.save("result/mask_gray.png")
#         print("생성된 마스크 이미지가 mask_output.png로 저장되었습니다.")
#     except Exception as e:
#         print(f"마스크 이미지 저장 오류: {e}")
        
#     return mask.resize((768, 1024))


#Densepose모델 추론(사람의 포즈 정보를 추출하여 이미지로 저장)
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
    pose_img_pil = Image.fromarray(pose_img).resize((768, 1024))
    # pose_img_pil.save("result/densepose_output.png")
    # print("DensePose 결과 이미지가 densepose_output.png로 저장되었습니다.")
    return pose_img_pil

#VTON 수행
def generate_tryon_image(
    pipe,
    human_image: Image.Image,
    garment_image: Image.Image,
    mask: Image.Image,
    garment_description: str,
    denoise_steps: int = 50,
    seed: int = None
) -> Image.Image:
    # Prepare images
    pose_img = prepare_pose_image(human_image)
    pose_tensor = tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)
    garm_tensor = tensor_transform(garment_image).unsqueeze(0).to(device, torch.float16)
    
    torch.cuda.empty_cache()
    
    # Pipe, unet_encoder GPu 로드
    pipe.to(device)
    pipe.unet_encoder.to(device)

    # Set generator
    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

    # Generate embeddings
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

    # Generate image
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

def process_tryon(
    human_image_path: str,
    garment_image_path: str,
    base_path: str,
    cloth_type: str,
    mask_image_path: str = None,
    garment_description: str = "",
    auto_mask: bool = True,
    crop_image: bool = True,
    denoise_steps: int = 50,
    seed: int = None,
) -> Image.Image:
    # Initialize models
    print(garment_description)
    pipe, parsing_model, openpose_model = initialize_models(base_path)
    

    # Load images
    human_img = Image.open(human_image_path).convert("RGB")
    garment_img = Image.open(garment_image_path).convert("RGB").resize((768, 1024))

    # Process human image
    if crop_image:
        human_img_processed, crop_coords = crop_and_resize_image(human_img)
    else:
        human_img_processed = human_img.resize((768, 1024))
        crop_coords = None

    # Generate or load mask
    if auto_mask:
        mask = generate_mask(parsing_model, openpose_model, human_img_processed, cloth_type)
    else:
        if mask_image_path is None:
            raise ValueError("Mask image path must be provided when auto_mask is False")
        mask = pil_to_binary_mask(Image.open(mask_image_path)).resize((768, 1024))

    # Generate try-on image
    result = generate_tryon_image(
        pipe,
        human_img_processed,
        garment_img,
        mask,
        garment_description,
        denoise_steps,
        seed
    )

    # If image was cropped, paste result back into original image
    if crop_image and crop_coords:
        left, top, right, bottom = crop_coords
        original_size = (right - left, bottom - top)
        result = result.resize(original_size)
        human_img.paste(result, (int(left), int(top)))
        return human_img

    return result

# Example usage
if __name__ == "__main__":
    description = garment_description_gemini()
    result = process_tryon(
        human_image_path='img/human/bb_10M24D177.jpg',
        garment_image_path='img/garment/pants2.jpg',
        base_path='yisol/IDM-VTON',
        cloth_type="pants",
        garment_description=description,
        auto_mask=True,
        crop_image=True,
        denoise_steps=40,
        seed=42
    )

    # Save result
    result.save('result.png')