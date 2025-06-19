# import os
# import spacy
# # from dotenv import load_dotenv
# from fastapi import FastAPI
# from qdrant_client import QdrantClient
# from routes import ImageSearch, TextSearch
# from transformers import CLIPProcessor, CLIPModel

# app = FastAPI()
# # load_dotenv()

# # ================================================================
# # The @app.on_event("startup") is a decorator that specifies a function
# # to run automatically when the application starts up (i.e., when you run the server).
# #
# # Why use it?
# # 1. To load data or initialize settings before the app starts accepting requests.
# # 2. To set up things like database connections or load machine learning models.
# # ================================================================

# @app.on_event("startup")
# async def startup__span():
#     # Load the Feature Extraction model
#     MODEL_ID = "openai/clip-vit-base-patch32"
#     # MODEL_ID = "./models/clip_model"
#     model = CLIPModel.from_pretrained(MODEL_ID)
#     processor = CLIPProcessor.from_pretrained(MODEL_ID)

#     # create a client from qdrantDB
#     app.client = QdrantClient(
#         url=os.environ["QDRANT_HOST"],
#         api_key=os.environ["QDRANT_API_KEY"]
#     )

#     # load spacy pipeline
#     app.nlp = spacy.load("en_core_web_sm")



# @app.on_event("shutdown")
# async def shutdown_span():
#     pass


# app.include_router(TextSearch.text_router)
# app.include_router(ImageSearch.image_router)




import os
import spacy
# from dotenv import load_dotenv
from fastapi import FastAPI
from qdrant_client import QdrantClient
from routes import ImageSearch, TextSearch
from transformers import CLIPProcessor, CLIPModel

# Fix cache permission issues
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

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
    # Load the Feature Extraction model
    MODEL_ID = "openai/clip-vit-base-patch32"
    
    try:
        print("🔄 Attempting to load CLIP model...")
        # Try PyTorch first with explicit cache settings
        model = CLIPModel.from_pretrained(
            MODEL_ID,
            cache_dir="/tmp/transformers_cache",
            force_download=True,  # Force fresh download
            local_files_only=False,
            trust_remote_code=True
        )
        processor = CLIPProcessor.from_pretrained(
            MODEL_ID,
            cache_dir="/tmp/transformers_cache",
            force_download=True,
            local_files_only=False
        )
        print("✅ PyTorch model loaded successfully!")
        
    except Exception as e:
        print(f"⚠️ PyTorch loading failed: {e}")
        print("🔄 Trying TensorFlow weights...")
        try:
            # Fallback to TensorFlow weights
            model = CLIPModel.from_pretrained(
                MODEL_ID,
                from_tf=True,
                cache_dir="/tmp/transformers_cache",
                force_download=True,
                local_files_only=False
            )
            processor = CLIPProcessor.from_pretrained(
                MODEL_ID,
                cache_dir="/tmp/transformers_cache",
                force_download=True,
                local_files_only=False
            )
            print("✅ TensorFlow model loaded successfully!")
            
        except Exception as e2:
            print(f"❌ Both methods failed: {e2}")
            raise e2
    
    # Store in app for access in routes
    app.model = model
    app.processor = processor
    
    # create a client from qdrantDB
    app.client = QdrantClient(
        url=os.environ["QDRANT_HOST"],
        api_key=os.environ["QDRANT_API_KEY"]
    )
    
    # load spacy pipeline
    app.nlp = spacy.load("en_core_web_sm")
    
    print("🎉 All components loaded successfully!")

@app.on_event("shutdown")
async def shutdown_span():
    pass

app.include_router(TextSearch.text_router)
app.include_router(ImageSearch.image_router)