from PIL import Image
from fastapi.responses import JSONResponse
from controllers import FeatureExtractionControllers, SimilaritySearchControllers
from fastapi import File, UploadFile, Form, APIRouter, Request

image_router = APIRouter(
    prefix="/api/v1/image",
    tags=["api_v1", "image"],
)

@image_router.post("/ImageSearchQuery")
async def image_search_query(
    request: Request,
    File: UploadFile = File(...),
    limit: int = Form(5),
    color: str = Form(None)
):
    """
    Endpoint to handle image search queries.

    Parameters:
    - image: The image file to be processed.
    - limit: The number of images to return in response.
    - color: Optional color filter for the search.

    Returns:
    - A JSON response with the search results.
    """

    # read the image
    image = Image.open(File.file)

    # get the CLIP model and processor
    model, processor = request.app.model, request.app.processor

    # Extract input features
    features_Extractor = FeatureExtractionControllers(
        request=request
    )

    # extract image features
    image_features = features_Extractor.extract_image_features(
        image=image,
        model=model,
        processor=processor
    )

    # Search
    urls_response = []
    search_object = SimilaritySearchControllers(
        request = request,
        image_features = image_features,
        limit = limit
    )

    urls_response = []
    if color:
        urls_response = search_object.color_filter_search(color = color)

    else:
        urls_response = search_object.simple_search()
    
    print(urls_response)
    print(f"----------------------------------------> {type(urls_response)}")
    return JSONResponse(
        status_code=200,
        content=urls_response
    )
