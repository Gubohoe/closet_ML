import os
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModelForMaskGeneration
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from typing import List, Any, Optional
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy

import sys
sys.append("<official ip-adpater git repository folder>")
from ip_adapter import IPAdapter

## Grounded SAM(Segment Anythiing) for extracting roi image
# grounded dino model
model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda"

dino_processor = AutoProcessor.from_pretrained(model_id)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# segment anything
segmenter_id = "facebook/sam-vit-base"

sam_model = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
sam_processor = AutoProcessor.from_pretrained(segmenter_id)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def predict_dino(text, image, box_threshold, text_threshold):
    inputs = dino_processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)

    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )
    results = results[0]
    return [results["boxes"][0].cpu().numpy()], [results["scores"][0].cpu().numpy()], [results["labels"][0]]


def annotate_dino(image_source, boxes, logits, pharases):
    origin_image = deepcopy(image)
    draw = ImageDraw.Draw(image)

    # boxes = boxes.cpu().numpy()
    # logits = logits.cpu().numpy()

    # Define colors (you can modify these)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # RGB format

    # Draw each detection
    for idx, (box, score, phrase) in enumerate(zip(boxes, logits, pharases)):
        # Get coordinates
        x1, y1, x2, y2 = box.astype(int)

        # Select color (cycling through the color list)
        color = colors[idx % len(colors)]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Prepare text
        text = f"{phrase}: {score:.2f}"

        # Draw text background
        text_bbox = draw.textbbox((x1, y1), text)
        draw.rectangle([text_bbox[0], text_bbox[1]-2, text_bbox[2], text_bbox[3]+2],
                      fill=color)

        # Draw text
        draw.text((x1, y1), text, fill=(255, 255, 255))
        break

    return origin_image, image


def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)
    return masks

# Make masking image based on the result of SAM
def draw_mask(mask, image, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([1.0, 1.0, 1.0])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil)), mask_image_pil


def detect(image, text_prompt, model, box_threshold = 0.3, text_threshold = 0.25):
    boxes, logits, pharases = predict_dino(
        text=text_prompt,
        image=image,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    origin_frame, annotated_frame = annotate_dino(image_source=image, boxes=boxes, logits=logits, pharases=pharases)
    return origin_frame, annotated_frame, boxes


def segment(
    image: Image.Image,
    boxes,
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None
) -> List:
    inputs = sam_processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = sam_model(**inputs)
    masks = sam_processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)
    return masks

################################################################################################################
#### demo execution
image_path = "/fxflow/junwkim/poc/diffusion_fashion/sample_kids.png"
image = Image.open(image_path).convert("RGB") 

origin_frame, annotated_image, boxes = detect(image, "upper body.", dino_model)
segment_masks = segment(origin_frame, [[boxes[0].tolist()]])

# visualize
masked_segmentation_image, mask_image_pil = draw_mask(segment_masks[0], np.array(origin_frame), False)
masked_segmentation_image2, mask_image_pil2 = draw_mask(segment_masks[0], np.array(origin_frame), True)
grounded_sam_image = Image.fromarray(masked_segmentation_image2)

image_grid([origin_frame, annotated_image, mask_image_pil, grounded_sam_image], 1, 4)
################################################################################################################

## Virtual Try on with roi image
# image prompting 방식을 통한 roi region에서 이미지 합성

base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "out_models/models/image_encoder"
ip_ckpt = "out_models/models/ip-adapter_sd15.bin"
device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)
ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

################################################################################################################
garment_image = Image.open("./m2m.png")
images = ip_model.generate(
    pil_image=garment_image,
    num_samples=4,
    num_inference_steps=50,
    seed=42,
    image=origin_frame,
    mask_image=mask_image_pil,
    strength=0.9
)
grid = image_grid(images, 1, 4)
display(grid)
################################################################################################################

print("I am NewJeans")
