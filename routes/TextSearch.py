from fastapi.responses import JSONResponse
from fastapi import Form, APIRouter, Request
from controllers import SemanticSearchControllers, FeatureExtractionControllers

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
    # Load the CLIP model and processor
    model, processor = request.app.model, request.app.processor

    # extract text features
    feature_extractor = FeatureExtractionControllers(
        request=request
    )
    text_features = feature_extractor.extract_text_features(
        description=description,
        model=model,
        processor=processor
    )

    # Semantic search
    search_object = SemanticSearchControllers(
        request=request,
        limit=limit
    )
    urls_response = await search_object.search(
        text_features=text_features
    )

    return JSONResponse(
        status_code=200,
        content=urls_response
    )
