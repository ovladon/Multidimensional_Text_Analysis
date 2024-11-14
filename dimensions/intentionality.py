# dimensions/Intentionality.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from transformers.pipelines import Pipeline
import re

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class Intentionality(BaseDimension):
    def __init__(self):
        super().__init__(name="Intentionality")
        self.intentional_verbs = self.load_intentional_verbs()
        self.purpose_phrases = self.load_purpose_phrases()
        self.lemmatizer = WordNetLemmatizer()

    def load_intentional_verbs(self) -> set:
        """
        Loads a predefined set of intentional verbs.
        """
        intentional_verbs = set([
            # Expanded list of intentional verbs
            "intend", "plan", "aim", "seek", "strive", "endeavor", "design",
            "target", "aspire", "commit", "dedicate", "focus", "pursue",
            "resolve", "determine", "vow", "pledge", "promise", "ensure",
            "guarantee", "establish", "create", "build", "develop", "construct",
            "implement", "execute", "administer", "manage", "oversee",
            "coordinate", "organize", "lead", "direct", "guide", "facilitate",
            "promote", "advocate", "support", "champion", "encourage", "foster",
            "nurture", "cultivate", "enhance", "improve", "upgrade", "strengthen",
            "bolster", "augment", "expand", "increase", "accelerate", "boost",
            "innovate", "transform", "change", "reduce", "must", "should", "will",
            "shall", "would", "could", "can", "want", "wish", "desire", "hope",
            "need", "require", "expect", "intend", "plan"
        ])
        return intentional_verbs

    def load_purpose_phrases(self) -> set:
        """
        Loads a predefined set of purpose-driven phrases.
        """
        purpose_phrases = set([
            # Expanded list of purpose phrases
            "in order to", "so as to", "so that", "with the aim of",
            "with the intention of", "with the purpose of", "aimed at",
            "for the purpose of", "to achieve", "to accomplish",
            "to ensure", "to guarantee", "to promote", "to support",
            "to develop", "to create", "to build", "to establish",
            "to implement", "to execute", "to manage", "to oversee",
            "to coordinate", "to organize", "to lead", "to direct",
            "to guide", "to facilitate", "to encourage", "to foster",
            "to nurture", "to cultivate", "to enhance", "to improve",
            "to upgrade", "to strengthen", "to expand", "to increase",
            "to accelerate", "to boost", "to innovate", "to transform",
            "to change", "to reduce", "to mitigate", "to address"
        ])
        return purpose_phrases

    def standard_method(self, text: str) -> dict:
        """
        Calculates the intentionality score based on heuristic linguistic analysis.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Intentionality': {'score': 0.0, 'error': False}}

            total_intentional_verbs = 0
            total_purpose_phrases = 0
            total_sentences = len(sentences)

            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                # Count intentional verbs
                intentional_count = sum(1 for word in words if self.lemmatizer.lemmatize(word, 'v') in self.intentional_verbs)
                total_intentional_verbs += intentional_count

                # Count purpose-driven phrases
                for phrase in self.purpose_phrases:
                    if phrase in sentence.lower():
                        total_purpose_phrases += 1

            # Calculate ratios
            intentional_ratio = total_intentional_verbs / total_sentences
            purpose_ratio = total_purpose_phrases / total_sentences

            # Compute intentionality score
            intentional_score = ((intentional_ratio * 0.6) + (purpose_ratio * 0.4)) * 100
            intentional_score = max(0.0, min(100.0, intentional_score))

            return {'Intentionality': {'score': intentional_score, 'error': False}}

        except Exception as e:
            return {'Intentionality': {'score': 0.0, 'error': True, 'error_message': str(e)}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the intentionality score using a text classification model.
        """
        try:
            result = model(text, candidate_labels=['intentional', 'unintentional'])

            if not result:
                return {'Intentionality': {'score': 0.0, 'error': False}}

            label = result.get('labels', [])[0].lower()
            score = result.get('scores', [])[0]

            if 'intentional' in label:
                intentionality_score = score * 100
            elif 'unintentional' in label:
                intentionality_score = (1 - score) * 100
            else:
                intentionality_score = 50  # Neutral if label is unrecognized

            intentionality_score = max(0.0, min(100.0, intentionality_score))

            return {'Intentionality': {'score': intentionality_score, 'error': False}}

        except Exception as e:
            return {'Intentionality': {'score': 0.0, 'error': True, 'error_message': str(e)}}

