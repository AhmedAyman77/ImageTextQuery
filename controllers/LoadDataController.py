import json
import clip
from .BaseController import BaseController
from transformers import CLIPProcessor, CLIPModel


class LoadData(BaseController):
    def __init__(self):
        super().__init__()
    
    def load_json_data(self, file_path: str):
        """
        Load JSON data from a file.
        """
        with open(file_path, 'r') as f:
            Data = json.load(f)
        
        return Data
    
    def images_feature_extraction_models(self):
        """
        Load the feature extraction models.
        """
        MODEL_PATH = "../../Feature_Extraction_Models/clip_model"
        model = CLIPModel.from_pretrained(MODEL_PATH)
        processor = CLIPProcessor.from_pretrained(MODEL_PATH)
        
        return model, processor
    
        
    def text_feature_extraction_models(self):
        """
        Load the feature extraction models.
        """

        model, _ = clip.load("ViT-B/32", device="cpu")

        return model