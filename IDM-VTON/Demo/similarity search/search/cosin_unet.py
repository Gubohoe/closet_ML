import torch
import sys
sys.path.append('./')
from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
import io
from typing import List
import numpy as np
from torchvision import transforms
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from diffusers import AutoencoderKL

app = FastAPI()

# Global variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'yisol/IDM-VTON'  # 모델 경로
collection_name = "garment_vectors"

# Initialize models
def initialize_models(model_path: str = model_path):
    """Initialize UNet and VAE models"""
    print(f"Initializing models from {model_path}")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tensor_transform = transforms.Compose([
            transforms.Resize((768, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # Initialize UNet encoder
        unet_encoder = UNet2DConditionModel_ref.from_pretrained(
            model_path,
            subfolder="unet_encoder",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).requires_grad_(False)

        # Initialize VAE
        vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.float16
        ).requires_grad_(False)

        return unet_encoder.to(device).eval(), vae.to(device).eval(), tensor_transform

    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        raise e

def extract_features(image: Image.Image, unet_encoder: torch.nn.Module, vae: AutoencoderKL,
                    tensor_transform: transforms.Compose, target_dims: int = 65536):
    """Extract features from image using UNet and VAE"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Prepare image tensor
        image_tensor = tensor_transform(image)
        mask = torch.zeros((1, image_tensor.shape[1], image_tensor.shape[2]))
        image_tensor = torch.cat([image_tensor, mask], dim=0)
        image_tensor = image_tensor.unsqueeze(0).to(device, torch.float16)

        timesteps = torch.zeros(1, device=device)
        encoder_hidden_states = torch.zeros(1, 77, 2048, device=device, dtype=torch.float16)

        with torch.no_grad():
            # Get UNet features
            _, garment_features = unet_encoder(
                image_tensor,
                timesteps,
                encoder_hidden_states,
                return_dict=False
            )

            # Process features
            feature_vectors = []
            for feat in garment_features:
                if len(feat.shape) == 2:
                    feat_mean = feat
                elif len(feat.shape) == 3:
                    feat_mean = feat.mean(dim=2)
                elif len(feat.shape) == 4:
                    feat_mean = feat.mean(dim=[2, 3])
                else:
                    continue
                feature_vectors.append(feat_mean.cpu())

            all_features = torch.cat(feature_vectors, dim=1)
            
            # VAE compression
            target_latent_size = int(np.sqrt(target_dims // 4))
            input_side_length = int(target_latent_size / 0.13025)
            needed_features = 3 * input_side_length * input_side_length
            features_rgb = all_features[:, :needed_features].reshape(1, 3, input_side_length, input_side_length)
            features_rgb = features_rgb.to(device, torch.float16)
            features_rgb = 2 * (features_rgb - features_rgb.min()) / (features_rgb.max() - features_rgb.min()) - 1
            
            latents = vae.encode(features_rgb).latent_dist.sample()
            features = latents.flatten().cpu().numpy()
            
            if len(features) > target_dims:
                features = features[:target_dims]
            
            return features

    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        raise e

def search_similar_items(client: QdrantClient, vector: np.ndarray, category: str, top_k: int = 3, score_threshold: float = 0.85):
    """Search for similar items in Qdrant with score threshold"""
    try:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="category",
                    match=MatchValue(value=category)
                )
            ]
        )
        
        # Search with larger limit to ensure enough results after filtering
        initial_limit = top_k * 3
        results = client.search(
            collection_name=collection_name,
            query_vector=vector.tolist(),
            limit=initial_limit,
            query_filter=search_filter,
            score_threshold=score_threshold
        )
        
        # Filter by score threshold and sort
        filtered_results = [
            result for result in results
            if result.score >= score_threshold
        ]
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        
        # Take top k results
        top_results = filtered_results[:top_k]
        
        # Return only the URLs
        return [result.payload["image_url"] for result in top_results]
        
    except Exception as e:
        print(f"Error searching similar items: {str(e)}")
        raise e

# Initialize models at startup
unet_encoder, vae, tensor_transform = initialize_models()
qdrant_client = QdrantClient(host="localhost", port=6333)

@app.post("/search-similar")
async def search_similar(
    garment_image: UploadFile = File(...),
    cloth_type: str = Form(...)
):
    try:
        # Load and process image
        image = Image.open(io.BytesIO(await garment_image.read())).convert("RGB")
        
        # Extract features
        features = extract_features(image, unet_encoder, vae, tensor_transform)
        
        # Search similar items
        similar_urls = search_similar_items(
            qdrant_client, 
            features, 
            cloth_type
        )
        
        return {"similar_items": similar_urls}
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
@app.post("/search-similar-full")
async def search_similar_full(
    top_image: UploadFile = File(...),
    bottom_image: UploadFile = File(...)
):
    try:
        # 상의 이미지 처리
        top_img = Image.open(io.BytesIO(await top_image.read())).convert("RGB")
        features_top = extract_features(top_img, unet_encoder, vae, tensor_transform)
        
        # 하의 이미지 처리
        bottom_img = Image.open(io.BytesIO(await bottom_image.read())).convert("RGB")
        features_bottom = extract_features(bottom_img, unet_encoder, vae, tensor_transform)
        
        # 각각 검색
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
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


