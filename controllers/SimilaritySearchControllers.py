import torch
from fastapi import Request
from qdrant_client.http import models
from .BaseControllers import BaseControllers
from .SearchControllers import SearchControllers

class SimilaritySearchControllers(BaseControllers, SearchControllers):
    def __init__(
        self,
        request: Request,
        limit: int,
        image_features: torch.tensor
    ):
        BaseControllers.__init__(self)
        SearchControllers.__init__(self, limit=limit, request=request)
        self.image_features = image_features

    def simple_search(self):
        search_res = self.client.search(
            collection_name="image_features",
            query_vector=self.image_features[0].tolist(),
            limit=self.limit
        )

        return self.url_response(search_res=search_res)

    def color_filter_search(self, color: str):
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

        return self.url_response(search_res=search_res)
