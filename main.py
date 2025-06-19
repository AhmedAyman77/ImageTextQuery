import os
import spacy
from dotenv import load_dotenv
from fastapi import FastAPI
from qdrant_client import QdrantClient
from routes import ImageSearch, TextSearch
from transformers import CLIPProcessor, CLIPModel

app = FastAPI()
load_dotenv()

# ================================================================
# The @app.on_event("startup") is a decorator that specifies a function
# to run automatically when the application starts up (i.e., when you run the server).
#
# Why use it?
# 1. To load data or initialize settings before the app starts accepting requests.
# 2. To set up things like database connections or load machine learning models.
# ================================================================

@app.on_event("startup")
async def startup__span():
    # Load the Feature Extraction model
    # MODEL_ID = "openai/clip-vit-base-patch32"
    MODEL_ID = "./models/clip_model"
    app.model = CLIPModel.from_pretrained(MODEL_ID)
    app.processor = CLIPProcessor.from_pretrained(MODEL_ID)
    
    # create a client from qdrantDB
    app.client = QdrantClient(
        url=os.environ["QDRANT_HOST"],
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=60.0
    )

    # load spacy pipeline
    app.nlp = spacy.load("en_core_web_sm")


@app.on_event("shutdown")
async def shutdown_span():
    pass


app.include_router(TextSearch.text_router)
app.include_router(ImageSearch.image_router)
