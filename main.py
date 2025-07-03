import os
import spacy
# import pymssql
import aioodbc
from fastapi import FastAPI
from qdrant_client import QdrantClient
from helpers.config import get_settings_object
from transformers import CLIPProcessor, CLIPModel
from routes import ImageSearch, TextSearch, SyncUpdatedData

app = FastAPI()

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
    # get settings object for accessing environment variables
    settings = get_settings_object()


    # Load the Feature Extraction model
    
    # Fix cache permission issues
    # os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
    # os.environ["HF_HOME"] = "/tmp/hf_cache"
    # os.environ["TRANSFORMERS_OFFLINE"] = "0"


    # MODEL_ID = "openai/clip-vit-base-patch32"
    # try:
    #     # Try PyTorch first with explicit cache settings
    #     model = CLIPModel.from_pretrained(
    #         MODEL_ID,
    #         cache_dir="/tmp/transformers_cache",
    #         force_download=True,  # Force fresh download
    #         local_files_only=False,
    #         trust_remote_code=True
    #     )
    #     processor = CLIPProcessor.from_pretrained(
    #         MODEL_ID,
    #         cache_dir="/tmp/transformers_cache",
    #         force_download=True,
    #         local_files_only=False
    #     )
    # except Exception as e:
    #     try:
    #         # Fallback to TensorFlow weights
    #         model = CLIPModel.from_pretrained(
    #             MODEL_ID,
    #             from_tf=True,
    #             cache_dir="/tmp/transformers_cache",
    #             force_download=True,
    #             local_files_only=False
    #         )
    #         processor = CLIPProcessor.from_pretrained(
    #             MODEL_ID,
    #             cache_dir="/tmp/transformers_cache",
    #             force_download=True,
    #             local_files_only=False
    #         )
    #     except Exception as e2:
    #         raise e2

    model = CLIPModel.from_pretrained(
        "./models/clip_model"
    )

    processor = CLIPProcessor.from_pretrained(
        "./models/clip_model"
    )

    # Store in app for access in routes
    app.model = model
    app.processor = processor


    # create a client from qdrantDB
    app.client = QdrantClient(
        url=settings.QDRANT_HOST,
        api_key=settings.QDRANT_API_KEY
    )


    # create a SQL server database client
    connection_string = (
        f"Driver={{ODBC Driver 17 for SQL Server}};"
        f"Server={settings.SQL_HOST},{settings.SQL_PORT};"
        f"Database={settings.SQL_DATABASE};"
        f"UID={settings.SQL_USER};"
        f"PWD={settings.SQL_PASSWORD};"
        f"TrustServerCertificate=yes;"
    )

    app.SQLDatabasePool = await aioodbc.create_pool(
        dsn=connection_string,
        autocommit=True
    )

    # load spacy pipeline
    app.nlp = spacy.load("en_core_web_sm")


@app.on_event("shutdown")
async def shutdown_span():
    app.client = None
    if app.SQLDatabasePool:
        app.SQLDatabasePool.close()
        await app.SQLDatabasePool.wait_closed()


app.include_router(TextSearch.text_router)
app.include_router(ImageSearch.image_router)
app.include_router(SyncUpdatedData.sync_data_router)
