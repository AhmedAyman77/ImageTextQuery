import torch
import torch.nn.functional as F
from PIL import Image
from .BaseController import BaseController
from transformers import CLIPProcessor, CLIPModel

class FeatureExtractionControllers(BaseController):
    def __init__(self):
        super().__init__()
    
    def process_image(
        self,
        image: Image,
        processor: CLIPProcessor
    ):
        input = processor(images=image, return_tensors="pt")["pixel_values"]

        return input
    
    def extract_image_features(
        self,
        input:torch.tensor,
        model: CLIPModel
    ):
        
        with torch.no_grad():
            image_features = model.get_image_features(input)

        # convert the features to unit vector
        # cosine similarity only care about the angle between the feature vectors
        image_features = F.normalize(image_features, dim=-1)

        return image_features
    

    def process_text(
        self,
        description: str,
        processor: CLIPProcessor
    ):
        input = processor(text=description, return_tensors="pt")["input_ids"]
        
        return input
    
    def extract_text_features(
        self,
        input:torch.tensor,
        model: CLIPModel
    ):
        with torch.no_grad():
            text_features = model.get_text_features(input)

        # convert the features to unit vector
        # cosine similarity only care about the angle between the feature vectors
        text_features = F.normalize(text_features, dim=-1)
        
        return text_features
    