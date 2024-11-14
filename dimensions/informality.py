# dimensions/informality.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import Pipeline
import re

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

class Informality(BaseDimension):
    def __init__(self):
        super().__init__(name="Informality")
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
            "hey", "cool", "gotta", "ain't", "wanna", "gonna", "sup", "yo", "dude", "bro",
            "cuz", "bruh", "nah", "yeah", "yep", "nope", "ok", "okay", "alright", "k", "u",
            "ur", "gr8", "thx", "plz", "thx", "luv", "lol", "lmao", "rofl", "omg", "wtf",
            "fml", "idk", "tbh", "smh", "af", "btw", "ikr", "imho", "brb", "ttyl", "hmu",
            "nvm", "omw", "wyd", "wtg", "bff", "bae", "lit", "dope", "sick", "chill",
            "epic", "fail", "sucks", "sick", "totes", "obvi", "obvious", "crazy", "mad",
            "ridiculous", "ridic", "legit", "goat", "fam", "squad", "savage", "salty",
            "throw shade", "tea", "receipts", "shade", "clap back", "slay", "fire",
            "lowkey", "highkey", "flex", "finesse", "gucci", "turnt", "turn up", "basic",
            "thirsty", "ratchet", "af", "woke", "yeet"
        ])
        return informal_words

    def standard_method(self, text: str) -> dict:
        """
        Calculates the informality score based on heuristic linguistic analysis.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Informality': {'score': 0.0, 'error': False}}
            
            total_formal = 0
            total_informal = 0
            total_contractions = 0
            total_words = 0
            
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                if not words:
                    continue
                
                total_words += len(words)
                
                # Count formal and informal words
                formal_count = sum(1 for word in words if word in self.formal_keywords)
                informal_count = sum(1 for word in words if word in self.informal_keywords)
                
                total_formal += formal_count
                total_informal += informal_count
                
                # Count contractions
                contractions_found = len(self.contraction_pattern.findall(sentence))
                total_contractions += contractions_found
            
            if total_words == 0:
                informality_score = 0.0
            else:
                # Calculate ratios
                informal_ratio = (total_informal + total_contractions) / total_words
                formal_ratio = total_formal / total_words
                
                # Compute informality score
                informality_score = (informal_ratio * 0.7) + ((1 - formal_ratio) * 0.3)
                informality_score *= 100
                informality_score = max(0.0, min(100.0, informality_score))
            
            return {'Informality': {'score': informality_score, 'error': False}}
        
        except Exception as e:
            return {'Informality': {'score': 0.0, 'error': True, 'error_message': str(e)}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the informality score using a text classification model.
        """
        try:
            predictions = model(text)

            if not predictions:
                return {'Informality': {'score': 0.0, 'error': False}}

            # Handle different possible output structures
            if isinstance(predictions, list):
                top_prediction = predictions[0]
            else:
                top_prediction = predictions

            label = top_prediction.get('label', '').upper()
            score = top_prediction.get('score', 0.0)

            if 'INFORMAL' in label:
                informality_score = score * 100
            elif 'FORMAL' in label:
                informality_score = (1 - score) * 100
            else:
                informality_score = 50  # Neutral if label is unrecognized

            informality_score = max(0.0, min(100.0, informality_score))

            return {'Informality': {'score': informality_score, 'error': False}}

        except Exception as e:
            return {'Informality': {'score': 0.0, 'error': True, 'error_message': str(e)}}

