from PIL import Image
import torch
from torchvision import transforms 
from transformers import AutoModelForImageSegmentation
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import io

app = FastAPI()

# 모델 초기화
model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
model.to('cuda')
model.eval()

# 이미지 변환 설정 
image_size = (1024, 1024)
transform_image = transforms.Compose([
   transforms.Resize(image_size),
   transforms.ToTensor(),
   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def process_image(image: Image.Image) -> Image.Image:
   # 이미지 전처리
   input_images = transform_image(image).unsqueeze(0).to('cuda')
   
   # inference
   with torch.no_grad():
       preds = model(input_images)[-1].sigmoid().cpu()
   
   # 마스크 생성 및 적용
   pred = preds[0].squeeze()
   pred_pil = transforms.ToPILImage()(pred)
   mask = pred_pil.resize(image.size)
   image.putalpha(mask)
   
   return image

@app.post("/upload")
async def remove_background(file: UploadFile = File(...)):
   try:
       # 이미지 읽기
       contents = await file.read()
       image = Image.open(io.BytesIO(contents)).convert("RGB")
       
       # 배경 제거 처리
       processed_image = process_image(image)
       
       # 메모리에서 바로 이미지 반환
       img_byte_arr = io.BytesIO()
       processed_image.save(img_byte_arr, format='PNG')
       img_byte_arr = img_byte_arr.getvalue()
       
       return Response(content=img_byte_arr, media_type="image/png")
       
   except Exception as e:
       return {
           "success": False,
           "error": str(e)
       }