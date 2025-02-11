import torch
import sys
sys.path.append('./')
from PIL import Image
import numpy as np
from pathlib import Path
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from torchvision import transforms
import os
import pickle
import json

#특징 추출
def extract_features(image: Image.Image, unet_encoder: torch.nn.Module,
                    tensor_transform: transforms.Compose, device: str = 'cuda'):
   
    print("Extracting features from image...")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 이미지를 텐서로 변환하고 정규화
        image_tensor = tensor_transform(image)
        
        # 이미지와 같은 크기의 마스크(0으로 채워진) 생성
        mask = torch.zeros((1, image_tensor.shape[1], image_tensor.shape[2]))
        
        # 이미지 텐서와 마스크를 결합 (채널 차원으로)
        image_tensor = torch.cat([image_tensor, mask], dim=0)
        
        # 배치 차원 추가하고 GPU로 이동, float16으로 변환
        image_tensor = image_tensor.unsqueeze(0).to(device, torch.float16)

        timesteps = torch.zeros(1, device=device)
        encoder_hidden_states = torch.zeros(1, 77, 2048, device=device, dtype=torch.float16)

        with torch.no_grad():
            _, garment_features = unet_encoder(
                image_tensor,
                timesteps,
                encoder_hidden_states,
                return_dict=False
            )

            # 특징 벡터 저장을 위한 리스트
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

            if not feature_vectors:
                raise ValueError("No features were successfully processed")
            
            #벡터 결합 및 반환
            all_features = torch.cat(feature_vectors, dim=1).numpy()
            return all_features.flatten()

    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        raise e

#모델 초기화
def initialize_model(model_path: str = 'yisol/IDM-VTON', device: str = 'cuda'):
    print(f"Initializing model from {model_path}")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tensor_transform = transforms.Compose([
            transforms.Resize((768, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        unet_encoder = UNet2DConditionModel_ref.from_pretrained(
            model_path,
            subfolder="unet_encoder",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).requires_grad_(False)

        return unet_encoder.to(device).eval(), tensor_transform

    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        raise e
    
#벡터와 메타데이터를 파일로 저장
def save_features(features: np.ndarray, image_path: str, save_dir: str, category: str):
    os.makedirs(save_dir, exist_ok=True)
    
    # 이미지 파일명에서 확장자 제거
    base_name = Path(image_path).stem
    
    # 벡터 저장 (.npy 형식)
    vector_path = os.path.join(save_dir, f"{base_name}.npy")
    np.save(vector_path, features)
    
    # 메타데이터 저장 (JSON 형식)
    metadata = {
        'image_path': str(image_path),
        'vector_path': vector_path,
        'category': category,
        'vector_shape': features.shape
    }
    
    metadata_path = os.path.join(save_dir, f"{base_name}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved features for {base_name}")
    return vector_path, metadata_path

#폴더 내의 모든 의류 이미지 처리
def process_garment_folder(folder_path: str, category: str, save_dir: str = 'vectors'):
    print(f"\nProcessing folder: {folder_path}")
    print(f"Category: {category}")

    try:
        # 모델 초기화
        unet_encoder, tensor_transform = initialize_model()

        # 이미지 파일 찾기
        image_paths = list(Path(folder_path).glob('*.jpg')) + list(Path(folder_path).glob('*.png'))
        print(f"Found {len(image_paths)} images in folder")
        
        # 결과 저장할 딕셔너리
        all_metadata = {}

        for img_path in image_paths:
            try:
                print(f"\nProcessing image: {img_path}")

                # 이미지 로드 및 특징 추출
                image = Image.open(str(img_path)).convert('RGB')
                features = extract_features(image, unet_encoder, tensor_transform)
                
                # 특징 저장
                vector_path, metadata_path = save_features(
                    features, 
                    str(img_path), 
                    os.path.join(save_dir, category),
                    category
                )
                
                # 메타데이터 수집
                with open(metadata_path, 'r') as f:
                    all_metadata[str(img_path)] = json.load(f)

            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
                continue

            torch.cuda.empty_cache()
        
        # 전체 메타데이터 저장
        metadata_file = os.path.join(save_dir, f"{category}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
            
        print(f"\nProcessed {len(all_metadata)} images successfully.")
        print(f"Metadata saved to: {metadata_file}")

    except Exception as e:
        print(f"Error in process_garment_folder: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        # 상의 이미지 처리
        print("\nProcessing upper clothes...")
        process_garment_folder(
            folder_path="IDM-VTON/Demo/upper_clothes", 
            category="upper",
            save_dir="vectors"  # 벡터 저장 디렉토리
        )
    except Exception as e:
        print(f"\nScript failed with error: {str(e)}")