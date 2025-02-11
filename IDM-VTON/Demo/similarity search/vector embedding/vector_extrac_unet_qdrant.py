import torch
import sys
from PIL import Image
sys.path.append('./')
import numpy as np
from pathlib import Path
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from torchvision import transforms
import os
import time
import firebase_admin
from firebase_admin import credentials, firestore, storage
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Tuple
from diffusers import AutoencoderKL

# Firebase global variables
db = None
bucket = None

#Google Firebase 서비스 초기화
def initialize_firebase(cred_path: str, storage_bucket: str):
    try:
        global db, bucket
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            'storageBucket': storage_bucket
        })
        db = firestore.client()
        bucket = storage.bucket()
        print("Firebase initialization successful")
    except Exception as e:
        print(f"Firebase initialization error: {str(e)}")
        raise e

#firebase Storage 이미지 업로드
def upload_to_storage(image_path: str, category: str) -> str:
    try:
        timestamp = int(time.time())
        blob_path = f"garments/{category}/{timestamp}_{os.path.basename(image_path)}"
        print(f"Storage path: {blob_path}")
        
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(image_path)
        
        # Generate URL
        blob.make_public()
        url = blob.public_url
        print(f"Uploaded image URL: {url}")
        return url
    except Exception as e:
        print(f"Storage upload error: {str(e)}")
        raise e

#Qdrant clinet 초기화 및 Collection 생성
def initialize_qdrant(collection_name: str, vector_size: int) -> QdrantClient:
    try:
        client = QdrantClient(host="localhost", port=6333)
        
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE, on_disk=True)
            )
            print(f"Created new collection: {collection_name}")
        else:
            print(f"Collection {collection_name} already exists")
            
        return client
    
    except Exception as e:
        print(f"Error initializing Qdrant: {str(e)}")
        raise e

#Qdrant 벡터 및 메타 데이터 저장
def save_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    features: np.ndarray,
    metadata: Dict[str, Any],
    point_id: int
) -> bool:
    try:
        # Debug info
        print("\nDebug info before saving:")
        print(f"Features type: {type(features)}")
        print(f"Features shape: {features.shape}")
        print(f"Features min/max: {features.min():.4f}/{features.max():.4f}")
        vector_list = features.tolist()
        print(f"Vector list length: {len(vector_list)}")
        print(f"First few values: {vector_list[:5]}")
        
        point = PointStruct(
            id=point_id,
            vector=vector_list,
            payload=metadata
        )
        
        # Debug point structure
        print("\nPoint structure:")
        print(f"Point ID: {point.id}")
        print(f"Vector present in point: {hasattr(point, 'vector')}")
        print(f"Vector length in point: {len(point.vector) if hasattr(point, 'vector') else 'N/A'}")
        
        # Perform upsert
        upsert_result = client.upsert(
            collection_name=collection_name,
            points=[point]
        )
        
        # Verify the save
        saved_point = client.retrieve(
            collection_name=collection_name,
            ids=[point_id]
        )
        
        print("\nVerification after save:")
        if saved_point:
            print(f"Point found in database: Yes")
            print(f"Vector present in saved point: {saved_point[0].vector is not None}")
            if saved_point[0].vector is not None:
                print(f"Saved vector length: {len(saved_point[0].vector)}")
        else:
            print("Point not found in database")
        
        print(f"Saved point {point_id} to Qdrant")
        return True
        
    except Exception as e:
        print(f"Error saving to Qdrant: {str(e)}")
        print(f"Error type: {type(e)}")
        return False

#모델 초기화
def initialize_model(model_path: str = 'yisol/IDM-VTON', device: str = 'cuda'):
    """Initialize both UNet and VAE models"""
    print(f"Initializing models from {model_path}")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tensor_transform = transforms.Compose([
            transforms.Resize((768, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # Initialize UNet encoder for garment feature extraction
        unet_encoder = UNet2DConditionModel_ref.from_pretrained(
            model_path,
            subfolder="unet_encoder",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).requires_grad_(False)

        # Initialize VAE for dimensionality reduction
        vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.float16
        ).requires_grad_(False)

        return unet_encoder.to(device).eval(), vae.to(device).eval(), tensor_transform

    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        raise e

#특징 추출
def extract_features(image: Image.Image, target_dims: int = 65536, device = "cuda"):
    # 이미지를 텐서로 변환하고 정규화
    image_tensor = tensor_transform(image)
    
    # 이미지와 같은 크기의 마스크(0으로 채워진) 생성
    mask = torch.zeros((1, image_tensor.shape[1], image_tensor.shape[2]))
    
    # 이미지 텐서와 마스크를 결합 (채널 차원으로)
    image_tensor = torch.cat([image_tensor, mask], dim=0)
    
    # 배치 차원 추가하고 GPU로 이동, float16으로 변환
    image_tensor = image_tensor.unsqueeze(0).to(device, torch.float16)

    # UNet 인코더에 필요한 시간 스텝과 인코더 상태 초기화
    timesteps = torch.zeros(1, device=device)
    encoder_hidden_states = torch.zeros(1, 77, 2048, device=device, dtype=torch.float16)

    with torch.no_grad():  # 그래디언트 계산 비활성화
        # UNet 인코더로 이미지 특징 추출
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
        latents = vae.encode(features_rgb).latent_dist.sample()
        features = latents.flatten().cpu().numpy()
        
        # 목표 차원 수에 맞게 자르기
        if len(features) > target_dims:
            features = features[:target_dims]
        
        return features

#폴더 내 이미지 벡터 추출 및 저장 Process
def process_garment_folder(
    folder_path: str,
    category: str,
    collection_name: str,
    unet_encoder: torch.nn.Module,
    vae: AutoencoderKL,
    tensor_transform: transforms.Compose,
    start_id: int = 0  # 시작 ID를 파라미터로 받음
):
    print(f"\nProcessing folder: {folder_path}")
    print(f"Category: {category}")
    print(f"Starting from ID: {start_id}")

    try:
        image_paths = list(Path(folder_path).glob('*.jpg')) + list(Path(folder_path).glob('*.png'))
        print(f"Found {len(image_paths)} images in folder")
        
        sample_image = Image.open(str(image_paths[0])).convert('RGB')
        sample_features = extract_features(sample_image, unet_encoder, vae, tensor_transform)
        vector_size = len(sample_features)
        client = initialize_qdrant(collection_name, vector_size)
        
        processed_count = 0
        current_id = start_id  # 시작 ID부터 시작
        
        for img_path in image_paths:
            try:
                print(f"\nProcessing image: {img_path} (ID: {current_id})")
                storage_url = upload_to_storage(str(img_path), category)
                image = Image.open(str(img_path)).convert('RGB')
                features = extract_features(image, unet_encoder, vae, tensor_transform)
                
                metadata = {
                    'image_url': storage_url,
                    'category': category,
                    'vector_shape': list(features.shape),
                    'original_path': str(img_path)
                }
                
                save_to_qdrant(client, collection_name, features, metadata, current_id)
                processed_count += 1
                current_id += 1  # ID 증가

            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
                continue

            torch.cuda.empty_cache()
            
        print(f"\nProcessed {processed_count} images successfully.")
        return current_id  # 다음 카테고리를 위해 마지막 ID 반환

    except Exception as e:
        print(f"Error in process_garment_folder: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        # Initialize Firebase
        cred_path = "path/to/your/firebase-credentials.json"
        storage_bucket = ""
        initialize_firebase(cred_path, storage_bucket)
        
        # Initialize models
        print("\nInitializing models...")
        unet_encoder, vae, tensor_transform = initialize_model()
        
        # Process images
        collection_name = "garment_vectors"
        
        # Define categories to process
        categories = [
            {
                "path": "IDM-VTON/Demo/upper_clothes",
                "category": "upper_body"
            },
            {
                "path": "IDM-VTON/Demo/lower_clothes",
                "category": "lower_body"
            },
            {
                "path": "IDM-VTON/Demo/jumpsuit",
                "category": "jumpsuit"
            }
        ]
        
        # Process each category with continued IDs
        next_id = 0
        for category_info in categories:
            print(f"\nProcessing {category_info['category']}...")
            next_id = process_garment_folder(
                folder_path=category_info['path'],
                category=category_info['category'],
                collection_name=collection_name,
                unet_encoder=unet_encoder,
                vae=vae,
                tensor_transform=tensor_transform,
                start_id=next_id  # 이전 카테고리에서 끝난 ID부터 시작
            )
            
        print("\nAll categories processed successfully!")
        print(f"Total processed IDs: {next_id}")
        
    except Exception as e:
        print(f"\nScript failed with error: {str(e)}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()