import torch
from fastapi import Request
from qdrant_client.http import models
from .BaseController import BaseController

class SimilaritySearchControllers(BaseController):
    def __init__(
        self,
        client: Request,
        image_features: torch.tensor,
        limit: int,
    ):
        super().__init__()
        self.client = client
        self.image_features = image_features
        self.limit = limit


    def simple_search(self):
        search_res = self.client.search(
            collection_name="image_features",
            query_vector=self.image_features[0].tolist(),
            limit=self.limit
        )

        urls_response = []

        for hit in search_res:
            curr_dict = {}
            IMAGE_URL = hit.payload["link"]
            curr_dict["image_url"] = IMAGE_URL
            urls_response.append(curr_dict)

        return urls_response


    def color_filter_search(
        self,
        color: str
    ):
        search_res = self.client.search(
            collection_name="image_features",
            query_vector=self.image_features[0].tolist(),
            limit=self.limit,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="color",
                        match=models.MatchValue(value=color)
                    )
                ]
            )
        )

        urls_response = []
        for hit in search_res:
            curr_dict = {}
            IMAGE_URL = hit.payload["link"]
            curr_dict["image_url"] = IMAGE_URL
            urls_response.append(curr_dict)
        
        return urls_response
