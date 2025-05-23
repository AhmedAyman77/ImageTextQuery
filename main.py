import os
from routes import search
from fastapi import FastAPI
from controllers import LoadData
from qdrant_client import QdrantClient
from helpers import get_settings_object


app = FastAPI()


@app.on_event("startup")
async def startup_event():
    # load data
    load_data = LoadData()

    # load feature extraction models
    app.Data = load_data.load_json_data(os.path.join(os.path.dirname(__file__), "data", "data.json"))
    app.image_model, app.processor = load_data.images_feature_extraction_models()
    app.text_model = load_data.text_feature_extraction_models()

    # connect to Qdrant
    settings = get_settings_object()
    app.client = QdrantClient(
        url=settings.QDRANT_HOST,
        api_key=settings.QDRANT_API_KEY,
        timeout=60.0
    )



app.include_router(search.search_router)
