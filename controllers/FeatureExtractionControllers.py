import torch
import clip
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from .BaseController import BaseController
from transformers import CLIPProcessor, CLIPModel
from fastapi import UploadFile


class FeatureExtraction(BaseController):
    def __init__(self):
        super().__init__()
    
    async def extract_image_features(
        self,
        model: CLIPModel,
        processor: CLIPProcessor,
        file: UploadFile
    ):
        """
        Extract features from the data using the given model and processor.
        """
            
        contents = await file.read()
        img = Image.open(BytesIO(contents))

        input = processor(images=img, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            query_image = model.get_image_features(input)

        query_image = F.normalize(query_image, dim=-1)
        return query_image.cpu().numpy()[0]
    

    def extract_text_features(
        self,
        model: clip,
        description: str
    ):
        """
        Extract features from the user description using the given model.
        """
        tokens = clip.tokenize([description]).to("cpu")
        with torch.no_grad():
            query_vec = model.encode_text(tokens)[0]
            query_vec = F.normalize(query_vec, p=2, dim=-1)
        
        return query_vec.tolist()
