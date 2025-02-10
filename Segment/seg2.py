from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import io
import numpy as np

app = FastAPI()

# 모델 초기화
device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(device)
model.eval()

# 이미지 변환 설정
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/upload")
async def remove_background(file: UploadFile = File(...)):
    try:
        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 이미지 전처리
        input_tensor = transform_image(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs.logits
            
            # 원본 크기로 업샘플링
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False,
            )
            
            # 마스크 얻기
            pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        # 의류 마스크 생성
        clothes_labels = [1, 4, 5, 6, 7, 8, 9, 10, 16, 17]
        clothes_mask = np.isin(pred_seg, clothes_labels)
        
        # RGBA로 변환하여 알파 채널 처리
        image_rgba = image.convert('RGBA')
        image_array = np.array(image_rgba)
        image_array[~clothes_mask, 3] = 0
        
        result_image = Image.fromarray(image_array)
        
        # 메모리에서 바로 이미지 반환
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return Response(content=img_byte_arr, media_type="image/png")
        
    except Exception as e:
        return {"success": False, "error": str(e)}

