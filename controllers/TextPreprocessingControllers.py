import re
from fastapi import Request
from .BaseController import BaseController

class TextPreProcessingControllers(BaseController):
    def __init__(self, request:Request):
        super().__init__()
        self.request = request

    def preprocess_text(
        self,
        text: str
    ):
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        doc = self.request.app.nlp(text.lower())
        tokens = [
            token.lemma_ for token in doc
        ]
        return " ".join(tokens)
