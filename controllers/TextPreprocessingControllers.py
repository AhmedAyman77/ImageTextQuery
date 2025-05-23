import spacy
from .BaseController import BaseController


class TextPreprocessing(BaseController):
    def __init__(self):
        super().__init__()
    
    def preprocess_text(self, text):
        """
        Use spaCy to preprocess the input text:
        - Lowercasing
        - Removing stopwords, punctuation, and non-alphabetic tokens
        - Lemmatization
        """
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text.lower())
        tokens = [
            token.lemma_ for token in doc
            if token.is_alpha and not token.is_stop
        ]
        return " ".join(tokens)
