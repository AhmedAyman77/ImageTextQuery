import torch
from fastapi import Request
from .BaseController import BaseController

class SemanticSearchControllers(BaseController):
    def __init__(self):
        super().__init__()

    def search(
        self,
        request: Request,
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
            idx = hit.id
            urls_response.append(request.app.furniture[idx])
        
        return urls_response
