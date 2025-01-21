from fastapi import FastAPI, File, UploadFile
from transformers import pipeline
from PIL import Image
import numpy as np
import io
import base64

app = FastAPI()
segmenter = pipeline(model="mattmdjaga/segformer_b2_clothes")

def segment_clothing(img, clothes=["Hat", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Scarf"]):
    segments = segmenter(img)
    mask_list = []
    for s in segments:
        if s['label'] in clothes:
            mask_list.append(s['mask'])

    if mask_list:
        final_mask = np.stack(mask_list, axis=0).sum(axis=0)
        final_mask = np.clip(final_mask, 0, 255).astype(np.uint8)
    else:
        final_mask = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)

    final_mask = Image.fromarray(final_mask)
    img.putalpha(final_mask)
    return img

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGBA")
        
        # 세그멘트 수행
        segmented_img = segment_clothing(img)
        
        # 이미지를 base64로 인코딩
        buffered = io.BytesIO()
        segmented_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "success": True,
            "image": img_str
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }