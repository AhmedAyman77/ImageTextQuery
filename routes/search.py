from fastapi.responses import JSONResponse
from controllers import FeatureExtraction, SimilaritySearch, TextPreprocessing
from helpers import get_settings_object, Settings
from fastapi import APIRouter, Depends, UploadFile, status, Request, Form, File


import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

search_router = APIRouter(
    prefix="/search",
    tags=["data"]
)

@search_router.post("/image")
async def search_similar_images(
    request: Request,
    k: int = Form(10),
    color: str = Form(None),
    file: UploadFile = File(...),
    app_settings: Settings = Depends(get_settings_object)
):
    # get loaded data
    Data = request.app.Data

    # get model and processor
    model, process = request.app.image_model, request.app.processor

    # get qdrant client
    client = request.app.client

    # get query image features
    feature_extraction = FeatureExtraction()
    query_image_features = await feature_extraction.extract_image_features(model, process, file)

    # Similarity Search
    similarity_search = SimilaritySearch()
    content = similarity_search.get_KMost_similar_images(
        k=k,
        Data=Data,
        color=color,
        query_image=query_image_features,
        client=client
    )

    return JSONResponse(
        content=content
    )



@search_router.post("/text")
async def search_images_by_text(
    request: Request,
    k: int = Form(10),
    description: str = Form(...),
    app_settings: Settings = Depends(get_settings_object)
):
    # get loaded data
    Data = request.app.Data

    # get feature extraction model
    model = request.app.text_model

    # get qdrant client
    client = request.app.client

    # text_preprocessing
    preprocess = TextPreprocessing()
    description = preprocess.preprocess_text(description)

    # get query text features
    feature_extraction = FeatureExtraction()
    query_vec = feature_extraction.extract_text_features(model, description)

    # Similarity Search
    similarity_search = SimilaritySearch()
    content = similarity_search.get_KMost_similar_images_by_text(
        k=k,
        Data=Data,
        query_vec=query_vec,
        client=client
    )

    return JSONResponse(
        content=content
    )



'''
    COLORS = [
    'gold', 'red', 'oak', 'pink', 'ivory', 'black', 'brass', 'brown', 'beige',
    'gray', 'blue', 'copper', 'silver', 'green', 'titanium', 'mauve', 'bronze',
    'azure', 'ochre', 'yellow', 'cream', 'sage', 'teal', 'purple', 'onyx',
    'coral', 'plum', 'orange', 'white', 'graphite', 'charcoal', 'burgundy',
    'marble', 'natural wood', 'chrome', 'navy', 'concrete', 'pearl', 'olive',
    'slate', 'ruby', 'emerald', 'crimson', 'walnut', 'sapphire', 'ash',
    'amber', 'taupe', 'blush', 'steel', 'terra', 'cognac', 'obsidian',
    'crystal', 'mint'
]
'''