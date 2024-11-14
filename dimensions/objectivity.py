# dimensions/objectivity.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize
from transformers import Pipeline
import re

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

class Objectivity(BaseDimension):
    def __init__(self):
        super().__init__(name="Objectivity")
        self.objective_words = self.load_objective_words()
        self.subjective_words = self.load_subjective_words()

    def load_objective_words(self) -> set:
        """
        Loads a comprehensive set of objective words.
        """
        objective_words = set([
            "data", "statistics", "evidence", "research", "study", "analysis",
            "fact", "objective", "measure", "quantify", "observe", "experiment",
            "calculate", "determine", "demonstrate", "indicate", "results",
            "findings", "report", "document", "record", "information",
            "figure", "table", "graph", "chart", "percentage", "probability",
            "frequency", "average", "median", "mode", "standard deviation",
            "variance", "distribution", "correlation", "causation", "hypothesis",
            "theory", "model", "algorithm", "method", "procedure", "protocol",
            "system", "framework", "structure", "function", "component",
            "parameter", "variable", "constant", "equation", "formula",
            "statistical", "empirical", "quantitative", "qualitative",
            "objective"
        ])
        return objective_words

    def load_subjective_words(self) -> set:
        """
        Loads a comprehensive set of subjective words.
        """
        subjective_words = set([
            "believe", "feel", "think", "seem", "appear", "suggest",
            "imagine", "assume", "guess", "estimate", "impression",
            "perhaps", "possibly", "probably", "maybe", "should",
            "could", "would", "might", "ought", "suppose", "hope",
            "wish", "desire", "need", "require", "expect", "intend",
            "plan", "doubt", "concern", "worry", "love",
            "hate", "prefer", "like", "dislike", "enjoy", "adore",
            "despise", "passionate", "enthusiastic", "excited",
            "nervous", "anxious", "happy", "sad", "angry", "frustrated",
            "annoyed", "bored", "interested", "curious", "eager", "keen",
            "reluctant", "hesitant", "disappointed", "satisfied", "proud",
            "ashamed", "guilty", "embarrassed"
        ])
        return subjective_words

    def standard_method(self, text: str) -> dict:
        """
        Calculates the objectivity score based on heuristic linguistic analysis.
        """
        try:
            words = word_tokenize(text.lower())
            total_words = len(words)
            if total_words == 0:
                return {'Objectivity': {'score': 0.0, 'error': False}}

            objective_count = sum(1 for word in words if word in self.objective_words or word.isdigit())
            subjective_count = sum(1 for word in words if word in self.subjective_words)

            # Compute net objectivity
            net_objectivity = objective_count - subjective_count
            # Normalize to 0-100 scale
            objectivity_score = ((net_objectivity / total_words) + 0.5) * 100
            objectivity_score = max(0.0, min(100.0, objectivity_score))

            return {'Objectivity': {'score': objectivity_score, 'error': False}}
        except Exception as e:
            return {'Objectivity': {'score': 0.0, 'error': True, 'error_message': str(e)}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the objectivity score using zero-shot classification.
        """
        try:
            result = model(text, candidate_labels=['objective', 'subjective'])

            if not result:
                return {'Objectivity': {'score': 0.0, 'error': False}}

            label = result.get('labels', [])[0].lower()
            score = result.get('scores', [])[0]

            if 'objective' in label:
                objectivity_score = score * 100
            elif 'subjective' in label:
                objectivity_score = (1 - score) * 100
            else:
                objectivity_score = 50  # Neutral if label is unrecognized

            objectivity_score = max(0.0, min(100.0, objectivity_score))

            return {'Objectivity': {'score': objectivity_score, 'error': False}}

        except Exception as e:
            return {'Objectivity': {'score': 0.0, 'error': True, 'error_message': str(e)}}

