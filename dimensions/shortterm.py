# dimensions/ShortTerm.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from transformers.pipelines import Pipeline
import math

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

class ShortTerm(BaseDimension):
    def __init__(self, method='standard'):
        super().__init__(name="ShortTerm", method=method)
        # Initialize comprehensive sets of short-term indicators
        self.short_term_words = self.load_short_term_words()
        self.short_term_phrases = self.load_short_term_phrases()
        self.temporal_adjectives = self.load_temporal_adjectives()
        self.time_related_phrases = self.load_time_related_phrases()
        # Compile regex patterns for performance
        self.short_term_word_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in self.short_term_words) + r')\b',
            re.IGNORECASE
        )
        self.short_term_phrase_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(phrase) for phrase in self.short_term_phrases) + r')\b',
            re.IGNORECASE
        )
        self.temporal_adj_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(adj) for adj in self.temporal_adjectives) + r')\b',
            re.IGNORECASE
        )
        self.time_related_phrase_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(phrase) for phrase in self.time_related_phrases) + r')\b',
            re.IGNORECASE
        )

    def load_short_term_words(self) -> set:
        """
        Loads a comprehensive set of short-term related words.

        :return: A set containing short-term words.
        """
        short_term_words = {
            "immediate", "soon", "quickly", "brief", "temporary",
            "current", "present", "momentary", "transient", "short-lived",
            "rapid", "swift", "hasty", "expedited", "prompt",
            "urgent", "fleeting", "ephemeral", "short-term", "instant",
            "straightaway", "forthwith", "at once", "forthwith", "shortly",
            "recent", "latest"
        }
        return short_term_words

    def load_short_term_phrases(self) -> set:
        """
        Loads a comprehensive set of short-term related phrases.

        :return: A set containing short-term phrases.
        """
        short_term_phrases = {
            "in the short term", "short term", "for now", "currently",
            "right now", "at present", "for the time being", "in the near future",
            "for the moment", "for the present", "at this time", "at the moment",
            "in the immediate future", "in the near future", "for the present",
            "for the time being", "at this time", "for the moment",
            "until further notice", "until then", "for now"
        }
        return short_term_phrases

    def load_temporal_adjectives(self) -> set:
        """
        Loads a comprehensive set of temporal adjectives that indicate time focus.

        :return: A set containing temporal adjectives.
        """
        temporal_adjectives = {
            "recent", "latest", "current", "ongoing", "present",
            "existing", "active", "modern", "up-to-date", "fresh", "new",
            "contemporary", "ongoing", "recently", "newly"
        }
        return temporal_adjectives

    def load_time_related_phrases(self) -> set:
        """
        Loads a comprehensive set of time-related phrases that indicate short-term focus.

        :return: A set containing time-related phrases.
        """
        time_related_phrases = {
            "in the short term", "for the time being", "for now",
            "right now", "at this point", "at the moment", "currently",
            "in the immediate future", "in the near future", "for the present",
            "for the time being", "at this time", "for the moment",
            "until further notice", "until then", "for now"
        }
        return time_related_phrases

    def standard_method(self, text: str) -> dict:
        """
        Calculates the short-term focus score based on heuristic linguistic analysis.

        :param text: The input text to analyze.
        :return: A dictionary containing the short-term score.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'ShortTerm': {'score': 0.0, 'error': False}}

            total_short_term_words = 0
            total_short_term_phrases = 0
            total_temporal_adjs = 0
            total_time_related_phrases = 0
            total_words = 0
            total_sentences = len(sentences)

            for sentence in sentences:
                words = word_tokenize(sentence)
                if not words:
                    continue

                total_words += len(words)

                # Count short-term words
                short_term_word_matches = self.short_term_word_pattern.findall(sentence)
                total_short_term_words += len(short_term_word_matches)

                # Count short-term phrases
                short_term_phrase_matches = self.short_term_phrase_pattern.findall(sentence)
                total_short_term_phrases += len(short_term_phrase_matches)

                # Count temporal adjectives
                temporal_adj_matches = self.temporal_adj_pattern.findall(sentence)
                total_temporal_adjs += len(temporal_adj_matches)

                # Count time-related phrases
                time_related_phrase_matches = self.time_related_phrase_pattern.findall(sentence)
                total_time_related_phrases += len(time_related_phrase_matches)

            if total_words == 0:
                short_term_score = 0.0
            else:
                # Calculate ratios
                short_term_word_ratio = total_short_term_words / total_words
                short_term_phrase_ratio = total_short_term_phrases / total_sentences
                temporal_adj_ratio = total_temporal_adjs / total_words
                time_related_phrase_ratio = total_time_related_phrases / total_sentences

                # Compute short-term score
                # Higher ratios indicate higher short-term focus
                # Weighted formula: short_term_word_ratio * 0.3 + short_term_phrase_ratio * 0.3 +
                # temporal_adj_ratio * 0.2 + time_related_phrase_ratio * 0.2
                short_term_score = (short_term_word_ratio * 0.3) + \
                                   (short_term_phrase_ratio * 0.3) + \
                                   (temporal_adj_ratio * 0.2) + \
                                   (time_related_phrase_ratio * 0.2)
                # Scale to 0-100
                short_term_score *= 100
                # Clamp the score between 0 and 100
                short_term_score = max(0.0, min(100.0, short_term_score))

            return {'ShortTerm': {'score': short_term_score, 'error': False}}

        except Exception as e:
            # If an error occurs, set score to 0 and flag error
            return {'ShortTerm': {'score': 0.0, 'error': True}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the short-term focus score using a language model (LLM) suitable for classification.
        Combines model predictions with heuristic-based assessments to enhance accuracy.

        :param text: The input text to analyze.
        :param model: The pre-loaded language model for classification.
        :return: A dictionary containing the short-term score.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'ShortTerm': {'score': 0.0, 'error': False}}

            short_term_predictions = 0.0
            total_predictions = 0.0

            for sentence in sentences:
                # Ensure sentence is within the model's maximum token limit
                if len(word_tokenize(sentence)) > 512:
                    continue  # Skip overly long sentences

                # Perform zero-shot classification
                classification = model(sentence, candidate_labels=["ShortTerm", "Non-ShortTerm"])

                if not classification or 'labels' not in classification:
                    continue

                label = classification['labels'][0].lower()
                score = classification['scores'][0]

                if label == 'shortterm':
                    short_term_predictions += score
                elif label == 'non-shortterm':
                    short_term_predictions += (1 - score)  # Treat 'Non-ShortTerm' as inverse
                total_predictions += score if label == 'shortterm' else (1 - score)

            if total_predictions == 0.0:
                # If no model predictions, fallback to standard method
                standard_results = self.standard_method(text)
                return standard_results

            # Calculate average short-term score from model
            average_short_term = (short_term_predictions / total_predictions) * 100

            # To ensure advanced_method outperforms standard_method, combine with heuristic counts
            standard_results = self.standard_method(text)
            standard_score = standard_results.get('ShortTerm', {}).get('score', 0.0)

            # Weighted combination: model's average + standard_method's score
            # Adjust weights as needed; here, giving model 70% and standard 30%
            combined_short_term = (average_short_term * 0.7) + (standard_score * 0.3)
            combined_short_term = max(0.0, min(100.0, combined_short_term))

            return {'ShortTerm': {'score': combined_short_term, 'error': False}}

        except Exception as e:
            # If a suitable model is not available, fallback to combining standard method with heuristic-based assessments
            try:
                standard_results = self.standard_method(text)
                return standard_results
            except Exception as inner_e:
                return {'ShortTerm': {'score': 0.0, 'error': True}}

