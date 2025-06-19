import torch
from fastapi import Request
from qdrant_client.http import models
from .BaseController import BaseController

class SemanticSearchControllers(BaseController):
    def __init__(self):
        super().__init__()

    def search(
        self,
        client: Request,
        text_features: torch.tensor,
        limit: int,
    ):
        search_res = client.search(
            collection_name = "text_features",
            query_vector = text_features[0].tolist(),
            limit = limit
        )

        urls_response = []

        for hit in search_res:
            curr_dict = {}
            IMAGE_URL = hit.payload["link"]
            ID = hit.payload["id"]
            curr_dict["image_url"] = IMAGE_URL
            curr_dict["id"] = ID
            urls_response.append(curr_dict)

        return urls_response
