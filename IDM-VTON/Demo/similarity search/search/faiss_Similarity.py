import firebase_admin
from firebase_admin import credentials, firestore
import faiss
import numpy as np
from typing import List, Dict, Tuple
import time
from Recommand import ImageEmbedding

class GarmentSearchService:
    def __init__(self, firebase_cred_path: str):
        """
        의류 이미지 검색 서비스 초기화
        
        Parameters
        ----------
        firebase_cred_path : str
            Firebase 인증 파일 경로
        """
        self._initialize_firebase(firebase_cred_path)
        self.index = None
        self.id_map = {}  # faiss 인덱스와 firestore 문서 ID 매핑
        self.url_map = {}  # firestore 문서 ID와 URL 매핑
        
    def _initialize_firebase(self, cred_path: str):
        """Firebase 초기화"""
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            print("Firebase 초기화 성공")
        except Exception as e:
            print(f"Firebase 초기화 오류: {str(e)}")
            raise e

    def load_vectors(self, collection_name: str = 'garments', category: str = None):
        """
        Firestore에서 벡터 데이터 로드 및 Faiss 인덱스 생성
        
        Parameters
        ----------
        collection_name : str
            Firestore 컬렉션 이름
        category : str, optional
            특정 카테고리만 로드할 경우 지정
        """
        try:
            print("벡터 데이터 로드 중...")
            # 쿼리 생성
            query = self.db.collection(collection_name)
            if category:
                query = query.where('category', '==', category)
            
            # 문서 가져오기
            docs = query.get()
            vectors = []
            doc_ids = []
            
            print("문서 처리 중...")
            for i, doc in enumerate(docs):
                data = doc.to_dict()
                vector = data.get('vector')
                url = data.get('url')
                
                if vector and url:
                    vectors.append(vector)
                    doc_ids.append(doc.id)
                    self.url_map[doc.id] = url
            
            if not vectors:
                print("로드된 벡터 없음")
                return
            
            # Numpy 배열로 변환
            vectors_np = np.array(vectors, dtype='float32')
            print(f"로드된 벡터 수: {len(vectors)}")
            print(f"벡터 차원: {vectors_np.shape[1]}")
            
            # Faiss 인덱스 생성
            self.index = faiss.IndexFlatL2(vectors_np.shape[1])
            self.index.add(vectors_np)
            
            # ID 매핑 저장
            self.id_map = {i: doc_id for i, doc_id in enumerate(doc_ids)}
            
            print("Faiss 인덱스 생성 완료")
            
        except Exception as e:
            print(f"벡터 로드 오류: {str(e)}")
            raise e

    def search_similar(self, query_vector: np.ndarray, k: int = 3) -> List[Dict[str, str]]:
        """
        유사한 이미지 검색
        
        Parameters
        ----------
        query_vector : np.ndarray
            검색할 쿼리 벡터
        k : int
            반환할 결과 수
            
        Returns
        -------
        List[Dict[str, str]]
            유사한 이미지들의 정보 (URL 등)
        """
        if self.index is None:
            raise ValueError("인덱스가 초기화되지 않았습니다. load_vectors()를 먼저 실행하세요.")
            
        try:
            # 벡터 형태 확인 및 변환
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
            query_vector = query_vector.astype('float32')
            
            # 유사도 검색 수행
            distances, indices = self.index.search(query_vector, k)
            
            # 결과 포맷팅
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= 0:  # -1은 결과가 없는 경우
                    doc_id = self.id_map.get(idx)
                    if doc_id:
                        results.append({
                            'url': self.url_map.get(doc_id),
                            'distance': float(dist),
                            'rank': i + 1
                        })
            
            return results
            
        except Exception as e:
            print(f"검색 오류: {str(e)}")
            raise e

    def search_by_image(self, image_service, image_path: str, k: int = 3) -> List[Dict[str, str]]:
        """
        이미지 파일로 유사한 이미지 검색
        
        Parameters
        ----------
        image_service : ImageEmbedding
            이미지 임베딩 서비스 인스턴스
        image_path : str
            검색할 이미지 경로
        k : int
            반환할 결과 수
            
        Returns
        -------
        List[Dict[str, str]]
            유사한 이미지들의 정보
        """
        try:
            # 이미지에서 벡터 추출
            query_vector = image_service.get_embeddings(image_path)
            
            # 유사도 검색 수행
            return self.search_similar(query_vector, k)
            
        except Exception as e:
            print(f"이미지 검색 오류: {str(e)}")
            raise e


# 사용 예시
if __name__ == "__main__":
    # 검색 서비스 초기화
    search_service = GarmentSearchService(
        firebase_cred_path="Demo/recommand-c51a9-firebase-adminsdk-fbsvc-ca4988c8ae.json"
    )
    
    # 벡터 데이터 로드
    search_service.load_vectors(category="upper")  # 특정 카테고리만 로드
    
    # 이미지로 검색하는 경우
    # 임베딩 서비스 초기화
    image_service = ImageEmbedding(
        model_name="resnet50",
        pooling="avg",
        target_size=(224, 224)
    )
    
    # 이미지로 검색
    results = search_service.search_by_image(
        image_service=image_service,
        image_path="img/garment/image.png",
        k=3
    )
    
    # 결과 출력
    print("\n유사한 이미지 검색 결과:")
    for result in results:
        print(f"순위: {result['rank']}")
        print(f"URL: {result['url']}")
        print(f"거리: {result['distance']:.4f}")
        print()
        