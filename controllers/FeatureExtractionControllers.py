import re
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import Request
from .BaseControllers import BaseControllers
from transformers import CLIPProcessor, CLIPModel

class FeatureExtractionControllers(BaseControllers):
    def __init__(self, request: Request):
        super().__init__()
        self.request = request
    
    def process_image(
        self,
        image: Image,
        processor: CLIPProcessor
    ):
        input = processor(images=image, return_tensors="pt")["pixel_values"]

        return input
    
    def extract_image_features(
        self,
        image: Image,
        model: CLIPModel,
        processor: CLIPProcessor
    ):
        input = self.process_image(image, processor)
        with torch.no_grad():
            image_features = model.get_image_features(input)

        # convert the features to unit vector
        # cosine similarity only care about the angle between the feature vectors
        image_features = F.normalize(image_features, dim=-1)

        return image_features

    def clean_text(
        self,
        text: str
    ):
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove non-alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        doc = self.request.app.nlp(text.lower())
        tokens = [
            token.lemma_
            for token in doc
        ]
        return " ".join(tokens)

    def text_processor(
        self,
        description: str,
        processor: CLIPProcessor
    ):
        description = self.clean_text(description)
        input = processor(text=description, return_tensors="pt")["input_ids"]
        
        return input
    
    def extract_text_features(
        self,
        description: str,
        model: CLIPModel,
        processor: CLIPProcessor
    ):
        input = self.text_processor(description, processor)
        with torch.no_grad():
            text_features = model.get_text_features(input)

        # convert the features to unit vector
        # cosine similarity only care about the angle between the feature vectors
        text_features = F.normalize(text_features, dim=-1)
        
        return text_features
