import numpy as np
from qdrant_client.http import models
from qdrant_client import QdrantClient
from .BaseController import BaseController


class SimilaritySearch(BaseController):
    def __init__(self):
        super().__init__()
    
    def get_KMost_similar_images(
        self,
        k: int,
        Data: dict,
        color: str,
        query_image: np.ndarray,
        client: QdrantClient
    ):
        if color:
            search_result = client.search(
                collection_name="images",
                query_vector=query_image,
                limit=k,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="color",
                            match=models.MatchValue(value=color)
                        )
                    ]
                )
            )
        
        else:
            search_result = client.search(
                collection_name="images",
                query_vector=query_image,
                limit=k
            )
        
        return [
            Data[res.id]["img_path"]
            for res in search_result
        ]
    
    def get_KMost_similar_images_by_text(
        self,
        k: int,
        Data: dict,
        query_vec: np.ndarray,
        client: QdrantClient
    ):
        search_result = client.search(
            collection_name="furniture_descriptions",
            query_vector=query_vec,
            limit=k
        )
        
        return [
            Data[res.id]["img_path"]
            for res in search_result
        ]
    