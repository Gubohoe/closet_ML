import sys
sys.path.append('./')
import torch
from PIL import Image, ImageDraw
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

# Global variables
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# def print_gpu_memory():
#     if torch.cuda.is_available():
#         print(f"GPU 총 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
#         print(f"할당된 메모리: {torch.cuda.memory_allocated(0) / 1024**2:.0f}MB")
#         print(f"캐시된 메모리: {torch.cuda.memory_reserved(0) / 1024**2:.0f}MB")
#         print(f"남은 메모리: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**2:.0f}MB")

def initialize_models(base_path: str) -> Tuple:
    """Initialize all required models and return them"""
    # Initialize UNet with float32
    unet = UNet2DConditionModel.from_pretrained(
        base_path,
        subfolder="unet",
        torch_dtype=torch.float32,
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

    # Initialize other models with float32
    text_encoder_one = CLIPTextModel.from_pretrained(
        base_path,
        subfolder="text_encoder",
        torch_dtype=torch.float32,
    ).requires_grad_(False)

    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float32,
    ).requires_grad_(False)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        base_path,
        subfolder="image_encoder",
        torch_dtype=torch.float32,
    ).requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(
        base_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).requires_grad_(False)

    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        base_path,
        subfolder="unet_encoder",
        torch_dtype=torch.float32,
    ).requires_grad_(False)

    # Initialize pipeline with float32
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
        torch_dtype=torch.float32,
    )
    pipe.unet_encoder = unet_encoder

    # Initialize parsing and pose models
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)

    return pipe, parsing_model, openpose_model

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
    mask_image = mask.resize((768, 1024))
    
    # 마스크 시각화를 위한 mask_gray 생성
    mask_gray = (1 - transforms.ToTensor()(mask_image)).unsqueeze(0) * tensor_transform(image)
    mask_gray = to_pil_image((mask_gray.squeeze().clamp(-1, 1) + 1) / 2)
     
    try:
        mask_image.save("result/mask_output.png")
        mask_gray.save("result/mask_gray.png")
        print("생성된 마스크 이미지가 mask_output.png로 저장되었습니다.")
    except Exception as e:
        print(f"마스크 이미지 저장 오류: {e}")
        
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
    pose_img_pil = Image.fromarray(pose_img).resize((768, 1024))
    pose_img_pil.save("result/densepose_output.png")
    print("DensePose 결과 이미지가 densepose_output.png로 저장되었습니다.")
    return pose_img_pil

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
    # 이미지 텐서는 GPU에서 생성
    pose_tensor = tensor_transform(pose_img).unsqueeze(0).to('cuda:0', torch.float16)
    garm_tensor = tensor_transform(garment_image).unsqueeze(0).to('cuda:0', torch.float16)
    
    torch.cuda.empty_cache()
    
    # pipe와 unet_encoder를 CPU로 이동하고 float32로 설정
    pipe.to('cpu')
    pipe.to(torch.float32)
    pipe.unet_encoder.to('cpu')
    pipe.unet_encoder.to(torch.float32)

    # CPU generator 사용
    generator = torch.Generator('cpu').manual_seed(seed) if seed is not None else None

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

    # Generate image using CPU
    images = pipe(
        prompt_embeds=prompt_embeds.to('cpu', torch.float32),
        negative_prompt_embeds=negative_prompt_embeds.to('cpu', torch.float32),
        pooled_prompt_embeds=pooled_prompt_embeds.to('cpu', torch.float32),
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to('cpu', torch.float32),
        num_inference_steps=denoise_steps,
        generator=generator,
        strength=1.0,
        pose_img=pose_tensor.to('cpu', torch.float32),
        text_embeds_cloth=prompt_embeds_c.to('cpu', torch.float32),
        cloth=garm_tensor.to('cpu', torch.float32),
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
    seed: int = None
) -> Image.Image:
    # Initialize models
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

if __name__ == "__main__":
    result = process_tryon(
        human_image_path='img/human/bb_10M24D177.jpg',
        garment_image_path='img/garment/pants2.jpg',
        base_path='yisol/IDM-VTON',
        cloth_type="lower_body",     #upper_body:상의, lower_body: 하의
        garment_description="navy blue cotton pants with playful orange fox pattern all over, elastic waistband and cropped wide-leg style",
        auto_mask=True,
        crop_image=True,
        denoise_steps=40,
        seed=42
    )

    # Save result
    result.save('result_pants.png')