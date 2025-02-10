import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.applications import ResNet50, InceptionV3, VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
import time
from pathlib import Path


class ImageEmbeddingService:
    def __init__(self, model_name="resnet50", pooling="avg", target_size=(224, 224),
                 firebase_cred_path=None, storage_bucket=None):
        """
        이미지 임베딩 및 Firebase 저장 서비스 초기화
        
        Parameters
        ----------
        model_name : str
            사용할 기본 모델 ('resnet50', 'inception_v3', 'vgg19')
        pooling : str
            풀링 방법 ('avg', 'max')
        target_size : tuple
            입력 이미지 크기
        firebase_cred_path : str
            Firebase 인증 파일 경로
        storage_bucket : str
            Firebase Storage 버킷 이름
        """
        # 임베딩 모델 초기화
        self.model_name = model_name.lower()
        self.pooling = pooling
        self.target_size = target_size
        self.model = self._build_model()
        
        # Firebase 초기화
        if firebase_cred_path and storage_bucket:
            self._initialize_firebase(firebase_cred_path, storage_bucket)
        
    def _initialize_firebase(self, cred_path, storage_bucket):
        """Firebase 서비스 초기화"""
        try:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {
                'storageBucket': storage_bucket
            })
            self.db = firestore.client()
            self.bucket = storage.bucket()
            print("Firebase 초기화 성공")
        except Exception as e:
            print(f"Firebase 초기화 오류: {str(e)}")
            raise e
        
    def _build_model(self):
        """임베딩 추출을 위한 모델 생성"""
        model_args = {
            'weights': 'imagenet',
            'include_top': False,
            'input_shape': (*self.target_size, 3)
        }
        
        if self.model_name == "resnet50":
            base_model = ResNet50(**model_args)
            self.preprocess = resnet_preprocess
        elif self.model_name == "inception_v3":
            base_model = InceptionV3(**model_args)
            self.preprocess = inception_preprocess
        elif self.model_name == "vgg19":
            base_model = VGG19(**model_args)
            self.preprocess = vgg_preprocess
        else:
            raise ValueError(f"지원하지 않는 모델: {self.model_name}")
            
        x = base_model.output
        if self.pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif self.pooling == 'max':
            x = GlobalMaxPooling2D()(x)
        else:
            raise ValueError(f"지원하지 않는 풀링 방법: {self.pooling}")
            
        return Model(inputs=base_model.input, outputs=x)
    
    def load_image(self, image_path):
        """단일 이미지 로드 및 전처리"""
        img = Image.open(image_path).convert('RGB')
        img = self._preprocess_image(img)
        return img
    
    def _preprocess_image(self, img):
        """이미지 전처리 - 리사이즈 및 패딩"""
        width, height = img.size
        if width == height:
            padded_img = img
        elif width > height:
            padded_img = Image.new(img.mode, (width, width), 0)
            padded_img.paste(img, (0, (width - height) // 2))
        else:
            padded_img = Image.new(img.mode, (height, height), 0)
            padded_img.paste(img, ((height - width) // 2, 0))
            
        resized_img = padded_img.resize(self.target_size)
        return np.array(resized_img)
    
    def get_embeddings(self, images):
        """이미지 배치에서 임베딩 추출"""
        if isinstance(images, str):
            images = [self.load_image(images)]
            images = np.array(images)
        elif isinstance(images, list):
            images = [self.load_image(img_path) for img_path in images]
            images = np.array(images)
            
        processed_images = self.preprocess(images)
        embeddings = self.model.predict(processed_images)
        return embeddings

    def upload_to_storage(self, image_path: str, category: str) -> str:
        """이미지를 Firebase Storage에 업로드하고 URL 반환"""
        try:
            timestamp = int(time.time())
            blob_path = f"garments/{category}/{timestamp}_{os.path.basename(image_path)}"
            print(f"Storage 경로: {blob_path}")
            
            blob = self.bucket.blob(blob_path)
            blob.upload_from_filename(image_path)
            
            # URL 생성
            blob.make_public()
            url = blob.public_url
            print(f"업로드된 이미지 URL: {url}")
            return url
        except Exception as e:
            print(f"Storage 업로드 오류: {str(e)}")
            raise e

    def save_to_firestore(self, url: str, vector: np.ndarray, category: str):
        """벡터와 메타데이터를 Firestore에 저장"""
        try:
            doc_ref = self.db.collection('garments').document()
            doc_data = {
                'url': url,
                'vector': vector.tolist(),
                'category': category,
                'timestamp': firestore.SERVER_TIMESTAMP
            }
            doc_ref.set(doc_data)
            print(f"Firestore 문서 ID: {doc_ref.id}")
            return doc_ref.id
        except Exception as e:
            print(f"Firestore 저장 오류: {str(e)}")
            raise e

    def process_folder(self, folder_path: str, category: str):
        """폴더 내의 모든 의류 이미지 처리 및 저장"""
        print(f"\n폴더 처리 시작: {folder_path}")
        print(f"카테고리: {category}")
        
        try:
            # 이미지 파일 찾기
            image_paths = list(Path(folder_path).glob('*.jpg')) + list(Path(folder_path).glob('*.png'))
            print(f"발견된 이미지 수: {len(image_paths)}")
            
            for img_path in image_paths:
                try:
                    print(f"\n이미지 처리 중: {img_path}")
                    
                    # 임베딩 추출
                    embeddings = self.get_embeddings(str(img_path))
                    print(f"임베딩 shape: {embeddings.shape}")
                    
                    # Storage에 업로드
                    image_url = self.upload_to_storage(str(img_path), category)
                    
                    # Firestore에 저장
                    doc_id = self.save_to_firestore(image_url, embeddings[0], category)
                    print(f"처리 완료: {doc_id}")
                    
                except Exception as e:
                    print(f"이미지 처리 오류 {img_path}: {str(e)}")
                    continue
                
        except Exception as e:
            print(f"폴더 처리 오류: {str(e)}")
            raise e


# 사용 예시
if __name__ == "__main__":
    # 서비스 초기화
    service = ImageEmbeddingService(
        model_name="resnet50",
        pooling="avg",
        target_size=(224, 224),
        firebase_cred_path="",
        storage_bucket=""
    )
    
    # # 단일 이미지 처리
    # embedding = service.get_embeddings("path/to/single/image.jpg")
    # print("단일 이미지 임베딩 shape:", embedding.shape)
    
    # 폴더 처리
    service.process_folder("IDM-VTON/Demo/clothes/upper_body", "upper")