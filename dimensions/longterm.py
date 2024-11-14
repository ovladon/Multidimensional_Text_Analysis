# dimensions/LongTerm.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers.pipelines import Pipeline
import re

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

class LongTerm(BaseDimension):
    def __init__(self):
        super().__init__(name="LongTerm")
        self.longterm_keywords = self.load_longterm_keywords()
        self.duration_phrases = self.load_duration_phrases()
        self.keyword_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in self.longterm_keywords) + r')\b', 
            re.IGNORECASE
        )
        self.duration_pattern = re.compile(
            r'(' + '|'.join(re.escape(phrase) for phrase in self.duration_phrases) + r')', 
            re.IGNORECASE
        )

    def load_longterm_keywords(self) -> set:
        """
        Loads a comprehensive set of long-term keywords.
        """
        longterm_keywords = set([
            "future", "long-term", "eventual", "ultimately", "sustainable",
            "foreseeable", "permanent", "lasting", "enduring", "persistent",
            "prolonged", "extended", "durable", "everlasting", "decades",
            "centuries", "millennia", "generations", "posterity",
            "forthcoming", "hereafter", "subsequent", "continuous",
            "longstanding", "upcoming", "lasting", "future generations"
        ])
        return longterm_keywords

    def load_duration_phrases(self) -> set:
        """
        Loads a comprehensive set of duration-related phrases.
        """
        duration_phrases = set([
            "in the future", "for years to come", "over the coming decades",
            "in the long run", "for the foreseeable future", "for future generations",
            "in the years ahead", "long-term", "sustainable development",
            "in the next decade", "in the coming years", "future prospects",
            "over time", "throughout the years", "lasting impact"
        ])
        return duration_phrases

    def standard_method(self, text: str) -> dict:
        """
        Calculates the long-term orientation score based on heuristic linguistic analysis.
        """
        try:
            sentences = sent_tokenize(text)
            total_sentences = len(sentences)
            if total_sentences == 0:
                return {'LongTerm': {'score': 0.0, 'error': False}}

            longterm_mentions = 0
            for sentence in sentences:
                keyword_matches = len(self.keyword_pattern.findall(sentence))
                duration_matches = len(self.duration_pattern.findall(sentence))
                longterm_mentions += keyword_matches + duration_matches

            longterm_score = (longterm_mentions / total_sentences) * 100
            longterm_score = max(0.0, min(100.0, longterm_score))

            return {'LongTerm': {'score': longterm_score, 'error': False}}
        except Exception as e:
            return {'LongTerm': {'score': 0.0, 'error': True, 'error_message': str(e)}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the long-term orientation score using zero-shot classification.
        """
        try:
            result = model(text, candidate_labels=['long-term', 'short-term'])

            if not result:
                return {'LongTerm': {'score': 0.0, 'error': False}}

            label = result.get('labels', [])[0].lower()
            score = result.get('scores', [])[0]

            if 'long-term' in label:
                longterm_score = score * 100
            elif 'short-term' in label:
                longterm_score = (1 - score) * 100
            else:
                longterm_score = 50  # Neutral if label is unrecognized

            longterm_score = max(0.0, min(100.0, longterm_score))

            return {'LongTerm': {'score': longterm_score, 'error': False}}

        except Exception as e:
            return {'LongTerm': {'score': 0.0, 'error': True, 'error_message': str(e)}}

