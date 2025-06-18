
import json
import torch
import torch.nn.functional as F
from qdrant_client import QdrantClient
from fastapi.responses import JSONResponse
from fastapi import Form, APIRouter, Request
from controllers import TextPreProcessingControllers, SemanticSearchControllers, FeatureExtractionControllers


text_router = APIRouter(
    prefix="/api/v1/text",
    tags=["api_v1", "text"],
)


@text_router.post("/TextSearchQuery")
async def text_search_query(
    request: Request,
    description: str = Form(...),
    limit: int = Form(5)
):
    """
    Endpoint to handle text search queries.
    
    Parameters:
    - description: The description of the furniture you desire.
    - limit: The number of images to return in response.
    
    Returns:
    - A JSON response with the search results.
    """
    # preprocess inout text
    processed_description = TextPreProcessingControllers(request=request).preprocess_text(description)
    
    # Load the CLIP model and processor
    model, processor = request.app.model, request.app.processor

    # Preprocess the text
    feature_extractor = FeatureExtractionControllers()
    input = feature_extractor.process_text(
        description = processed_description,
        processor = processor
    )
    
    # extract text features
    text_features = feature_extractor.extract_text_features(
        input = input,
        model = model
    )

    # qdrantDB client
    client = request.app.client

    # Semantic search
    search_object = SemanticSearchControllers()
    urls_response = search_object.search(
        client=client,
        text_features=text_features,
        limit=limit
    )

    return JSONResponse(
        status_code=200,
        content=urls_response
    )
