# dimensions/Static.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from transformers.pipelines import Pipeline
import math

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

class Static(BaseDimension):
    def __init__(self, method='standard'):
        super().__init__(name="Static", method=method)
        # Initialize comprehensive sets of static indicators
        self.static_words = self.load_static_words()
        self.static_phrases = self.load_static_phrases()
        self.detailed_descriptors = self.load_detailed_descriptors()
        self.formal_language = self.load_formal_language()
        # Compile regex patterns for performance
        self.static_word_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in self.static_words) + r')\b', 
            re.IGNORECASE
        )
        self.static_phrase_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(phrase) for phrase in self.static_phrases) + r')\b', 
            re.IGNORECASE
        )
        self.detailed_descriptor_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(descriptor) for descriptor in self.detailed_descriptors) + r')\b', 
            re.IGNORECASE
        )
        self.formal_language_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in self.formal_language) + r')\b', 
            re.IGNORECASE
        )

    def load_static_words(self) -> set:
        """
        Loads a comprehensive set of words related to stability and static content.

        :return: A set containing static-related words.
        """
        static_words = {
            "stable", "unchanging", "consistent", "fixed", "constant",
            "permanent", "enduring", "persistent", "steady", "reliable",
            "unwavering", "solid", "durable", "immutable", "persistent",
            "invariable", "fixed", "firm", "unvarying", "unfluctuating",
            "steady-state", "static", "fixed-rate", "fixed-term"
        }
        return static_words

    def load_static_phrases(self) -> set:
        """
        Loads a comprehensive set of phrases related to stability and static content.

        :return: A set containing static-related phrases.
        """
        static_phrases = {
            "unchanging over time", "consistent performance", "fixed structure",
            "constant rate", "permanent fixture", "enduring stability",
            "persistent condition", "steady progress", "reliable system",
            "unwavering commitment", "solid foundation", "durable materials",
            "immutable laws", "persistent effort", "invariable trend",
            "fixed pattern", "firm resolve", "unvarying schedule",
            "unfluctuating data", "steady-state", "static equilibrium",
            "fixed-rate", "fixed-term", "long-term stability",
            "steady-state conditions", "constant environment"
        }
        return static_phrases

    def load_detailed_descriptors(self) -> set:
        """
        Loads a comprehensive set of detailed descriptors that indicate static content.

        :return: A set containing detailed descriptors.
        """
        detailed_descriptors = {
            "temperature", "pressure", "volume", "density", "mass",
            "distance", "length", "width", "height", "time", "duration",
            "rate", "frequency", "speed", "velocity", "acceleration",
            "force", "energy", "power", "capacity", "size", "scale",
            "capacity", "voltage", "current", "resistance", "frequency",
            "intensity", "magnitude", "amplitude", "angle", "area",
            "perimeter", "circumference", "radius", "diameter", "volume",
            "surface area", "tangent", "integral", "derivative", "vector",
            "matrix", "tensor", "coordinate", "axis", "dimension"
        }
        return detailed_descriptors

    def load_formal_language(self) -> set:
        """
        Loads a comprehensive set of formal language indicators.

        :return: A set containing formal language words.
        """
        formal_language = {
            "therefore", "thus", "hence", "consequently", "furthermore",
            "moreover", "in addition", "more so", "as a result",
            "subsequently", "accordingly", "nevertheless", "notwithstanding",
            "regardless", "accordingly", "thereby", "henceforth",
            "notably", "significantly", "predominantly", "primarily",
            "notably", "significantly", "predominantly", "primarily"
        }
        return formal_language

    def standard_method(self, text: str) -> dict:
        """
        Calculates the static score based on heuristic linguistic analysis.

        :param text: The input text to analyze.
        :return: A dictionary containing the static score.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Static': {'score': 0.0, 'error': False}}

            total_static_words = 0
            total_static_phrases = 0
            total_detailed_descriptors = 0
            total_formal_language = 0
            total_words = 0
            total_sentences = len(sentences)

            for sentence in sentences:
                words = word_tokenize(sentence)
                if not words:
                    continue

                total_words += len(words)

                # Count static words
                static_word_matches = self.static_word_pattern.findall(sentence)
                total_static_words += len(static_word_matches)

                # Count static phrases
                static_phrase_matches = self.static_phrase_pattern.findall(sentence)
                total_static_phrases += len(static_phrase_matches)

                # Count detailed descriptors
                detailed_descriptor_matches = self.detailed_descriptor_pattern.findall(sentence)
                total_detailed_descriptors += len(detailed_descriptor_matches)

                # Count formal language
                formal_language_matches = self.formal_language_pattern.findall(sentence)
                total_formal_language += len(formal_language_matches)

            if total_words == 0:
                static_score = 0.0
            else:
                # Calculate ratios
                static_word_ratio = total_static_words / total_words
                static_phrase_ratio = total_static_phrases / total_sentences
                detailed_descriptor_ratio = total_detailed_descriptors / total_words
                formal_language_ratio = total_formal_language / total_words

                # Compute static score
                # Weighted formula: static_word_ratio * 0.25 + static_phrase_ratio * 0.25 +
                # detailed_descriptor_ratio * 0.25 + formal_language_ratio * 0.25
                static_score = (static_word_ratio * 0.25) + \
                               (static_phrase_ratio * 0.25) + \
                               (detailed_descriptor_ratio * 0.25) + \
                               (formal_language_ratio * 0.25)
                # Scale to 0-100
                static_score *= 100
                # Clamp the score between 0 and 100
                static_score = max(0.0, min(100.0, static_score))

            return {'Static': {'score': static_score, 'error': False}}

        except Exception as e:
            # If an error occurs, set score to 0 and flag error
            return {'Static': {'score': 0.0, 'error': True}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the static score using a language model (LLM) suitable for classification.
        Combines model predictions with heuristic-based assessments to enhance accuracy.

        :param text: The input text to analyze.
        :param model: The pre-loaded language model for classification.
        :return: A dictionary containing the static score.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Static': {'score': 0.0, 'error': False}}

            static_predictions = 0.0
            total_predictions = 0.0

            for sentence in sentences:
                # Ensure sentence is within the model's maximum token limit
                if len(word_tokenize(sentence)) > 512:
                    continue  # Skip overly long sentences

                # Perform zero-shot classification
                classification = model(sentence, candidate_labels=["Static", "Non-Static"])

                if not classification or 'labels' not in classification:
                    continue

                label = classification['labels'][0].lower()
                score = classification['scores'][0]

                if label == 'static':
                    static_predictions += score
                elif label == 'non-static':
                    static_predictions += (1 - score)  # Treat 'Non-Static' as inverse
                total_predictions += score if label == 'static' else (1 - score)

            if total_predictions == 0.0:
                # If no model predictions, fallback to standard method
                standard_results = self.standard_method(text)
                return standard_results

            # Calculate average static score from model
            average_static = (static_predictions / total_predictions) * 100

            # To ensure advanced_method outperforms standard_method, combine with heuristic counts
            standard_results = self.standard_method(text)
            standard_score = standard_results.get('Static', {}).get('score', 0.0)

            # Weighted combination: model's average + standard_method's score
            # Adjust weights as needed; here, giving model 70% and standard 30%
            combined_static = (average_static * 0.7) + (standard_score * 0.3)
            combined_static = max(0.0, min(100.0, combined_static))

            return {'Static': {'score': combined_static, 'error': False}}

        except Exception as e:
            # If a suitable model is not available, fallback to combining standard method with heuristic-based assessments
            try:
                standard_results = self.standard_method(text)
                return standard_results
            except Exception as inner_e:
                return {'Static': {'score': 0.0, 'error': True}}

