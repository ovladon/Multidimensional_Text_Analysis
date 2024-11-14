# dimensions/formality.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize
from transformers import Pipeline
import re

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

class Formality(BaseDimension):
    def __init__(self):
        super().__init__(name="Formality")
        self.formal_keywords = self.load_formal_keywords()
        self.informal_keywords = self.load_informal_keywords()
        self.contraction_pattern = re.compile(r"\b(?:can't|won't|n't|'re|'ve|'ll|'d|'s)\b", re.IGNORECASE)

    def load_formal_keywords(self) -> set:
        """
        Loads a comprehensive set of formal words.
        """
        formal_words = set([
            "purchase", "assist", "utilize", "endeavor", "commence", "terminate", "facilitate",
            "conduct", "demonstrate", "consequently", "subsequently", "approximately", "regardless",
            "therefore", "henceforth", "notwithstanding", "whereas", "implement", "modify",
            "enhance", "ameliorate", "acquire", "constitute", "procure", "ascertain", "elucidate",
            "elaborate", "depict", "illustrate", "evaluate", "analyze", "synthesize", "integrate",
            "formulate", "design", "strategize", "execute", "administer", "coordinate", "oversee",
            "supervise", "manage", "authorize", "approve", "allocate", "delegate", "spearhead",
            "champion", "navigate", "mediate", "optimize", "streamline", "in conclusion",
            "attended", "symposium", "herewith", "hereby", "henceforth", "heretofore",
            "aforementioned", "aforestated", "aforenoted", "proceed", "endeavor", "commence",
            "determine", "illustrate", "significant", "subsequent", "consequent", "prior",
            "previous", "furthermore", "moreover", "nevertheless", "nonetheless", "notwithstanding"
        ])
        return formal_words

    def load_informal_keywords(self) -> set:
        """
        Loads a comprehensive set of informal words and slang.
        """
        informal_words = set([
            "buy", "help", "use", "try", "start", "end", "make", "do", "show", "basically", "so",
            "like", "actually", "just", "gotta", "wanna", "gonna", "kinda", "sorta", "really",
            "totally", "literally", "cool", "awesome", "bad", "sucks", "bummer", "heck", "darn",
            "idiot", "silly", "funny", "crazy", "mad", "hate", "love", "fun", "chill", "hang out",
            "yolo", "lit", "on fleek", "bae", "smh", "tbh", "idk", "btw", "lmao", "rofl", "ttyl",
            "brb", "omg", "lol", "wtf", "fml", "hey", "cool", "gotta", "ain't", "wanna", "gonna",
            "sup", "yo", "dude", "bro", "cuz", "bruh", "nah", "yeah", "yep", "nope",
            "ok", "okay", "alright", "k", "u", "ur", "gr8", "thx", "plz", "thx", "luv"
        ])
        return informal_words

    def standard_method(self, text: str) -> dict:
        """
        Calculates the formality score based on heuristic linguistic analysis.
        """
        try:
            words = word_tokenize(text.lower())
            total_words = len(words)
            if total_words == 0:
                return {'Formality': {'score': 0.0, 'error': False}}

            formal_count = sum(1 for word in words if word in self.formal_keywords)
            informal_count = sum(1 for word in words if word in self.informal_keywords)
            contraction_count = len(self.contraction_pattern.findall(text.lower()))

            # Compute formality score
            # Higher formal count and lower informal and contraction counts indicate higher formality
            formality_ratio = (formal_count + (total_words - informal_count - contraction_count)) / (2 * total_words)
            formality_score = formality_ratio * 100
            formality_score = max(0.0, min(100.0, formality_score))

            return {'Formality': {'score': formality_score, 'error': False}}
        except Exception as e:
            return {'Formality': {'score': 0.0, 'error': True, 'error_message': str(e)}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the formality score using a text classification model.
        """
        try:
            predictions = model(text)

            if not predictions:
                return {'Formality': {'score': 0.0, 'error': False}}

            # Handle different possible output structures
            if isinstance(predictions, list):
                top_prediction = predictions[0]
            else:
                top_prediction = predictions

            label = top_prediction.get('label', '').upper()
            score = top_prediction.get('score', 0.0)

            if 'FORMAL' in label:
                formality_score = score * 100
            elif 'INFORMAL' in label:
                formality_score = (1 - score) * 100
            else:
                formality_score = 50  # Neutral if label is unrecognized

            formality_score = max(0.0, min(100.0, formality_score))

            return {'Formality': {'score': formality_score, 'error': False}}

        except Exception as e:
            return {'Formality': {'score': 0.0, 'error': True, 'error_message': str(e)}}

