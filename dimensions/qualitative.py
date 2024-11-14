# dimensions/Qualitative.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class Qualitative(BaseDimension):
    def __init__(self):
        super().__init__(name="Qualitative")
        self.qualitative_adjectives = set(["comprehensive", "detailed", "in-depth", "extensive", "thorough",
            "rich", "nuanced", "elaborate", "descriptive", "vivid",
            "intricate", "complex", "refined", "sophisticated", "profound",
            "substantial", "meaningful", "insightful", "expressive", "creative",
            "artistic", "reflective", "thoughtful", "analytical", "critical",
            "evaluative", "interpretative", "holistic", "systematic",
            "methodical", "organized", "structured", "coherent", "logical",
            "persuasive", "compelling", "convincing", "influential",
            "impactful", "perspicacious", "perceptive", "astute", "sharp",
            "keen", "acute", "intelligent", "intuitive", "innovative",
            "original", "unique", "distinctive", "exceptional", "remarkable",
            "extraordinary", "noteworthy", "significant", "important",
            "crucial", "vital", "essential", "fundamental", "paramount",
            "key", "principal", "major", "minor", "average", "median",
            "mode", "maximum", "minimum", "higher", "lower", "best",
            "worst", "primary", "secondary", "tertiary", "quantitative",
            "qualitative", "statistical", "analytic", "analytical", "empirical",
            "theoretical", "predictive", "descriptive", "exploratory",
            "confirmatory", "correlational", "causal", "comparative",
            "longitudinal", "cross-sectional"])

    def standard_method(self, text: str) -> dict:
        try:
            words = word_tokenize(text.lower())
            total_words = len(words)
            qualitative_count = sum(1 for word in words if word in self.qualitative_adjectives)
            qualitative_score = (qualitative_count / total_words) * 100 if total_words else 0.0
            return {'Qualitative': {'score': qualitative_score, 'error': False}}
        except Exception as e:
            return {'Qualitative': {'score': 0.0, 'error': True, 'error_message': str(e)}}


'''
# dimensions/qualitative.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from transformers.pipelines import Pipeline

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

class Qualitative(BaseDimension):
    def __init__(self):
        super().__init__(name="Qualitative")
        self.qualitative_adjectives = self.load_qualitative_adjectives()
        self.subjective_phrases = self.load_subjective_phrases()
        self.narrative_markers = self.load_narrative_markers()
        # Compile regex patterns
        self.qualitative_adj_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, self.qualitative_adjectives)) + r')\b', re.IGNORECASE)
        self.subjective_phrase_pattern = re.compile('|'.join(map(re.escape, self.subjective_phrases)), re.IGNORECASE)
        self.narrative_marker_pattern = re.compile('|'.join(map(re.escape, self.narrative_markers)), re.IGNORECASE)

    def load_qualitative_adjectives(self) -> set:
        """
        Loads qualitative adjectives.
        """
        qualitative_adjectives = {
            "comprehensive", "detailed", "in-depth", "extensive", "thorough",
            "rich", "nuanced", "elaborate", "descriptive", "vivid",
            "intricate", "complex", "refined", "sophisticated", "profound",
            "substantial", "meaningful", "insightful", "expressive", "creative",
            "artistic", "reflective", "thoughtful", "analytical", "critical",
            "evaluative", "interpretative", "holistic", "systematic",
            "methodical", "organized", "structured", "coherent", "logical",
            "persuasive", "compelling", "convincing", "influential",
            "impactful", "perspicacious", "perceptive", "astute", "sharp",
            "keen", "acute", "intelligent", "intuitive", "innovative",
            "original", "unique", "distinctive", "exceptional", "remarkable",
            "extraordinary", "noteworthy", "significant", "important",
            "crucial", "vital", "essential", "fundamental", "paramount",
            "key", "principal", "major", "minor", "average", "median",
            "mode", "maximum", "minimum", "higher", "lower", "best",
            "worst", "primary", "secondary", "tertiary", "quantitative",
            "qualitative", "statistical", "analytic", "analytical", "empirical",
            "theoretical", "predictive", "descriptive", "exploratory",
            "confirmatory", "correlational", "causal", "comparative",
            "longitudinal", "cross-sectional"
        }
        return qualitative_adjectives

    def load_subjective_phrases(self) -> set:
        """
        Loads a comprehensive set of subjective phrases that indicate personal opinions or biases.

        :return: A set containing subjective phrases.
        """
        subjective_phrases = {
            "in my opinion", "I believe", "I think", "I feel", "it seems",
            "I assume", "I suppose", "I guess", "I imagine", "I reckon",
            "from my perspective", "as I see it", "to my mind",
            "if you ask me", "I contend", "I argue", "I maintain",
            "I posit", "I assert", "I declare", "I propose", "I suggest",
            "I recommend", "I advocate", "I support", "I oppose",
            "I doubt", "I question", "I wonder", "I’m convinced",
            "I’m not sure", "I’m uncertain", "I’m doubtful", "I’m skeptical",
            "I have reservations", "I have concerns", "I have doubts"
        }
        return subjective_phrases

    def load_narrative_markers(self) -> set:
        """
        Loads a comprehensive set of narrative markers that indicate storytelling or descriptive elements.

        :return: A set containing narrative markers.
        """
        narrative_markers = {
            "once upon a time", "in the beginning", "there was", "there were",
            "as a child", "during my childhood", "I remember", "I recall",
            "back in the day", "at the time", "then", "later", "afterwards",
            "meanwhile", "simultaneously", "in the meantime", "subsequently",
            "eventually", "finally", "in conclusion", "to sum up",
            "ultimately", "hence", "therefore", "thus", "consequently",
            "accordingly", "as a result", "because of", "due to",
            "owing to", "since", "given that", "considering that",
            "in light of", "in view of", "on account of", "for the reason that",
            "with the help of", "thanks to", "by virtue of"
        }
        return narrative_markers

    def standard_method(self, text: str) -> dict:
        """
        Calculates the qualitative score based on heuristic linguistic analysis.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Qualitative': {'score': 0.0, 'error': False}}

            total_qualitative_adjs = 0
            total_subjective_phrases = 0
            total_narrative_markers = 0
            total_modals = 0
            total_words = 0

            # Define modal verbs
            modal_verbs = {"could", "would", "might", "may", "can", "shall", "should"}
            modal_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, modal_verbs)) + r')\b', re.IGNORECASE)

            for sentence in sentences:
                words = word_tokenize(sentence)
                if not words:
                    continue

                total_words += len(words)

                # Count matches
                total_qualitative_adjs += len(self.qualitative_adj_pattern.findall(sentence))
                total_subjective_phrases += len(self.subjective_phrase_pattern.findall(sentence))
                total_narrative_markers += len(self.narrative_marker_pattern.findall(sentence))
                total_modals += len(modal_pattern.findall(sentence))

            if total_words == 0:
                qualitative_score = 0.0
            else:
                # Calculate ratios
                qualitative_adj_ratio = total_qualitative_adjs / total_words
                subjective_phrase_ratio = total_subjective_phrases / len(sentences)
                narrative_marker_ratio = total_narrative_markers / len(sentences)
                modal_ratio = total_modals / len(sentences)

                # Compute qualitative score
                qualitative_score = (
                    (qualitative_adj_ratio * 0.4) +
                    (subjective_phrase_ratio * 0.2) +
                    (narrative_marker_ratio * 0.2) +
                    (modal_ratio * 0.2)
                ) * 100
                qualitative_score = max(0.0, min(100.0, qualitative_score))

            return {'Qualitative': {'score': qualitative_score, 'error': False}}

        except Exception as e:
            return {'Qualitative': {'score': 0.0, 'error': True, 'error_message': str(e)}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the qualitative score using a text classification model.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Qualitative': {'score': 0.0, 'error': False}}

            qualitative_predictions = 0.0
            total_predictions = 0

            for sentence in sentences:
                predictions = model(sentence)

                if not predictions:
                    continue

                top_prediction = predictions[0]
                label = top_prediction['label'].lower()
                score = top_prediction['score']

                if 'qualitative' in label:
                    qualitative_predictions += score
                elif 'non-qualitative' in label:
                    qualitative_predictions += (1 - score)
                total_predictions += 1  # Corrected increment

            if total_predictions == 0:
                average_qualitative = 0.0
            else:
                average_qualitative = (qualitative_predictions / total_predictions) * 100
                average_qualitative = max(0.0, min(100.0, average_qualitative))

            return {'Qualitative': {'score': average_qualitative, 'error': False}}

        except Exception as e:
            return {'Qualitative': {'score': 0.0, 'error': True, 'error_message': str(e)}}
'''


