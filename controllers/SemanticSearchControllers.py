import torch
from fastapi import Request
from .BaseControllers import BaseControllers
from .SearchControllers import SearchControllers

class SemanticSearchControllers(BaseControllers, SearchControllers):
    def __init__(
        self,
        request: Request,
        limit: int
    ):
        BaseControllers.__init__(self)
        SearchControllers.__init__(self, limit=limit, request=request)
    
    async def search(
        self,
        text_features: torch.tensor
    ):
        search_res = self.client.search(
            collection_name = "text_features",
            query_vector = text_features[0].tolist(),
            limit = self.limit
        )

        return await self.url_response(search_res=search_res)
