# dimensions/Time.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from transformers.pipelines import Pipeline
import math

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

class Time(BaseDimension):
    def __init__(self, method='standard'):
        super().__init__(name="Time", method=method)
        # Initialize comprehensive sets of temporal indicators
        self.temporal_words = self.load_temporal_words()
        self.temporal_phrases = self.load_temporal_phrases()
        self.temporal_adverbs = self.load_temporal_adverbs()
        self.time_related_nouns = self.load_time_related_nouns()
        # Compile regex patterns for performance
        self.temporal_word_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in self.temporal_words) + r')\b', 
            re.IGNORECASE
        )
        self.temporal_phrase_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(phrase) for phrase in self.temporal_phrases) + r')\b', 
            re.IGNORECASE
        )
        self.temporal_adverb_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(adverb) for adverb in self.temporal_adverbs) + r')\b', 
            re.IGNORECASE
        )
        self.time_related_noun_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(noun) for noun in self.time_related_nouns) + r')\b', 
            re.IGNORECASE
        )

    def load_temporal_words(self) -> set:
        """
        Loads a comprehensive set of time-related words.

        :return: A set containing temporal words.
        """
        temporal_words = {
            "now", "today", "yesterday", "tomorrow", "recently", "soon",
            "later", "early", "late", "future", "past", "present",
            "immediately", "eventually", "currently", "previously",
            "ultimately", "meanwhile", "subsequently", "shortly",
            "recent", "ongoing", "temporarily", "momentarily",
            "sometime", "sometimes", "daily", "weekly", "monthly",
            "yearly", "annually", "decade", "century", "millennium",
            "hourly", "minute", "second", "daily", "weekly", "monthly",
            "yearly", "quarterly", "biweekly", "biennial", "biennially",
            "decades", "centuries", "millennia", "hour", "minute", "second"
        }
        return temporal_words

    def load_temporal_phrases(self) -> set:
        """
        Loads a comprehensive set of temporal phrases.

        :return: A set containing temporal phrases.
        """
        temporal_phrases = {
            "in the morning", "in the afternoon", "in the evening",
            "in the past", "in the future", "at the moment",
            "for the time being", "in the near future", "over the years",
            "throughout the day", "during the week", "by the end of the month",
            "within the next year", "since last year", "up until now",
            "for the next few days", "in the long run", "in the short term",
            "as of now", "from now on", "before the meeting",
            "after the event", "during the process", "over the last decade",
            "in recent times", "in the last few years", "from this point forward"
        }
        return temporal_phrases

    def load_temporal_adverbs(self) -> set:
        """
        Loads a comprehensive set of temporal adverbs.

        :return: A set containing temporal adverbs.
        """
        temporal_adverbs = {
            "now", "yesterday", "today", "tomorrow", "soon", "later",
            "early", "late", "currently", "previously", "ultimately",
            "eventually", "shortly", "recently", "momentarily",
            "temporarily", "immediately", "sometime", "sometimes",
            "daily", "weekly", "monthly", "yearly", "hourly",
            "biweekly", "biennial", "biennially"
        }
        return temporal_adverbs

    def load_time_related_nouns(self) -> set:
        """
        Loads a comprehensive set of time-related nouns.

        :return: A set containing time-related nouns.
        """
        time_related_nouns = {
            "time", "moment", "hour", "minute", "second", "day",
            "week", "month", "year", "decade", "century", "millennium",
            "period", "era", "epoch", "interval", "duration", "timeline",
            "schedule", "agenda", "deadline", "future", "past",
            "present", "chronology", "sequence", "timeline", "milestone"
        }
        return time_related_nouns

    def standard_method(self, text: str) -> dict:
        """
        Calculates the temporal score based on heuristic linguistic analysis.

        :param text: The input text to analyze.
        :return: A dictionary containing the temporal score.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Time': {'score': 0.0, 'error': False}}

            total_temporal_words = 0
            total_temporal_phrases = 0
            total_temporal_adverbs = 0
            total_time_related_nouns = 0
            total_words = 0
            total_sentences = len(sentences)

            for sentence in sentences:
                words = word_tokenize(sentence)
                if not words:
                    continue

                total_words += len(words)

                # Count temporal words
                temporal_word_matches = self.temporal_word_pattern.findall(sentence)
                total_temporal_words += len(temporal_word_matches)

                # Count temporal phrases
                temporal_phrase_matches = self.temporal_phrase_pattern.findall(sentence)
                total_temporal_phrases += len(temporal_phrase_matches)

                # Count temporal adverbs
                temporal_adverb_matches = self.temporal_adverb_pattern.findall(sentence)
                total_temporal_adverbs += len(temporal_adverb_matches)

                # Count time-related nouns
                time_related_noun_matches = self.time_related_noun_pattern.findall(sentence)
                total_time_related_nouns += len(time_related_noun_matches)

            if total_words == 0:
                temporal_score = 0.0
            else:
                # Calculate ratios
                temporal_word_ratio = total_temporal_words / total_words
                temporal_phrase_ratio = total_temporal_phrases / total_sentences
                temporal_adverb_ratio = total_temporal_adverbs / total_words
                time_related_noun_ratio = total_time_related_nouns / total_words

                # Compute temporal score
                # Weighted formula: temporal_word_ratio * 0.25 + temporal_phrase_ratio * 0.25 +
                # temporal_adverb_ratio * 0.25 + time_related_noun_ratio * 0.25
                temporal_score = (temporal_word_ratio * 0.25) + \
                                 (temporal_phrase_ratio * 0.25) + \
                                 (temporal_adverb_ratio * 0.25) + \
                                 (time_related_noun_ratio * 0.25)
                # Scale to 0-100
                temporal_score *= 100
                # Clamp the score between 0 and 100
                temporal_score = max(0.0, min(100.0, temporal_score))

            return {'Time': {'score': temporal_score, 'error': False}}

        except Exception as e:
            # If an error occurs, set score to 0 and flag error
            return {'Time': {'score': 0.0, 'error': True}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the temporal score using a language model (LLM) suitable for classification.
        Combines model predictions with heuristic-based assessments to enhance accuracy.

        :param text: The input text to analyze.
        :param model: The pre-loaded language model for classification.
        :return: A dictionary containing the temporal score.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Time': {'score': 0.0, 'error': False}}

            temporal_predictions = 0.0
            total_predictions = 0.0

            for sentence in sentences:
                # Ensure sentence is within the model's maximum token limit
                if len(word_tokenize(sentence)) > 512:
                    continue  # Skip overly long sentences

                # Perform zero-shot classification
                classification = model(sentence, candidate_labels=["Temporal", "Non-Temporal"])

                if not classification or 'labels' not in classification:
                    continue

                label = classification['labels'][0].lower()
                score = classification['scores'][0]

                if label == 'temporal':
                    temporal_predictions += score
                elif label == 'non-temporal':
                    temporal_predictions += (1 - score)  # Treat 'Non-Temporal' as inverse
                total_predictions += score if label == 'temporal' else (1 - score)

            if total_predictions == 0.0:
                # If no model predictions, fallback to standard method
                standard_results = self.standard_method(text)
                return standard_results

            # Calculate average temporal score from model
            average_temporal = (temporal_predictions / total_predictions) * 100

            # To ensure advanced_method outperforms standard_method, combine with heuristic counts
            standard_results = self.standard_method(text)
            standard_score = standard_results.get('Time', {}).get('score', 0.0)

            # Weighted combination: model's average + standard_method's score
            # Adjust weights as needed; here, giving model 70% and standard 30%
            combined_temporal = (average_temporal * 0.7) + (standard_score * 0.3)
            combined_temporal = max(0.0, min(100.0, combined_temporal))

            return {'Time': {'score': combined_temporal, 'error': False}}

        except Exception as e:
            # If a suitable model is not available, fallback to combining standard method with heuristic-based assessments
            try:
                standard_results = self.standard_method(text)
                return standard_results
            except Exception as inner_e:
                return {'Time': {'score': 0.0, 'error': True}}

