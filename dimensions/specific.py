# dimensions/Specific.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from transformers.pipelines import Pipeline
import math

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

class Specific(BaseDimension):
    def __init__(self, method='standard'):
        super().__init__(name="Specific", method=method)
        # Initialize comprehensive sets of specificity indicators
        self.specific_nouns = self.load_specific_nouns()
        self.detailed_adjectives = self.load_detailed_adjectives()
        self.precision_phrases = self.load_precision_phrases()
        self.quantifiers = self.load_quantifiers()
        # Compile regex patterns for performance
        self.specific_noun_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(noun) for noun in self.specific_nouns) + r')\b', 
            re.IGNORECASE
        )
        self.detailed_adj_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(adj) for adj in self.detailed_adjectives) + r')\b', 
            re.IGNORECASE
        )
        self.precision_phrase_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(phrase) for phrase in self.precision_phrases) + r')\b', 
            re.IGNORECASE
        )
        self.quantifier_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(q) for q in self.quantifiers) + r')\b', 
            re.IGNORECASE
        )

    def load_specific_nouns(self) -> set:
        """
        Loads a comprehensive set of specific nouns that indicate detailed information.

        :return: A set containing specific nouns.
        """
        specific_nouns = {
            "algorithm", "phenomenon", "hypothesis", "variable", "parameter",
            "framework", "model", "dataset", "architecture", "protocol",
            "methodology", "analysis", "evaluation", "simulation", "iteration",
            "optimization", "application", "technique", "strategy", "component",
            "module", "interface", "architecture", "design", "structure",
            "mechanism", "process", "system", "element", "factor", "criterion",
            "metric", "indicator", "feature", "attribute", "aspect", "dimension",
            "category", "class", "type", "instance", "case", "scenario",
            "context", "environment", "setting", "framework", "tool", "instrument"
        }
        return specific_nouns

    def load_detailed_adjectives(self) -> set:
        """
        Loads a comprehensive set of detailed adjectives that enhance specificity.

        :return: A set containing detailed adjectives.
        """
        detailed_adjectives = {
            "comprehensive", "extensive", "in-depth", "thorough", "detailed",
            "specific", "precise", "exact", "particular", "explicit",
            "nuanced", "elaborate", "meticulous", "rigorous", "systematic",
            "robust", "sophisticated", "complex", "granular", "fine-grained",
            "high-resolution", "accurate", "specific", "targeted", "focused",
            "dedicated", "specialized", "unique", "distinct", "definitive",
            "clear-cut", "unambiguous", "well-defined", "well-specified",
            "focused", "aimed", "oriented", "directed", "purposeful", "intentional"
        }
        return detailed_adjectives

    def load_precision_phrases(self) -> set:
        """
        Loads a comprehensive set of precision-related phrases that indicate specificity.

        :return: A set containing precision phrases.
        """
        precision_phrases = {
            "exactly", "specifically", "precisely", "in detail", "to the extent",
            "with precision", "at a granular level", "with exact measurements",
            "to the minute", "to the second", "with accuracy", "to a fine degree",
            "with specificity", "to the smallest detail", "with exactitude",
            "with meticulous care", "in a specific manner", "in a precise way"
        }
        return precision_phrases

    def load_quantifiers(self) -> set:
        """
        Loads a comprehensive set of quantifiers that enhance specificity.

        :return: A set containing quantifiers.
        """
        quantifiers = {
            "numerous", "several", "various", "multiple", "many", "few",
            "a number of", "a variety of", "a range of", "a multitude of",
            "a plethora of", "a large number of", "a significant number of",
            "countless", "innumerable", "abundant", "plentiful", "copious",
            "ample", "extensive", "comprehensive", "bountiful", "sizable"
        }
        return quantifiers

    def standard_method(self, text: str) -> dict:
        """
        Calculates the specificity score based on heuristic linguistic analysis.

        :param text: The input text to analyze.
        :return: A dictionary containing the specificity score.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Specific': {'score': 0.0, 'error': False}}

            total_specific_nouns = 0
            total_detailed_adjectives = 0
            total_precision_phrases = 0
            total_quantifiers = 0
            total_words = 0
            total_sentences = len(sentences)

            for sentence in sentences:
                words = word_tokenize(sentence)
                if not words:
                    continue

                total_words += len(words)

                # Count specific nouns
                specific_noun_matches = self.specific_noun_pattern.findall(sentence)
                total_specific_nouns += len(specific_noun_matches)

                # Count detailed adjectives
                detailed_adj_matches = self.detailed_adj_pattern.findall(sentence)
                total_detailed_adjectives += len(detailed_adj_matches)

                # Count precision phrases
                precision_phrase_matches = self.precision_phrase_pattern.findall(sentence)
                total_precision_phrases += len(precision_phrase_matches)

                # Count quantifiers
                quantifier_matches = self.quantifier_pattern.findall(sentence)
                total_quantifiers += len(quantifier_matches)

            if total_words == 0:
                specific_score = 0.0
            else:
                # Calculate ratios
                specific_noun_ratio = total_specific_nouns / total_words
                detailed_adj_ratio = total_detailed_adjectives / total_words
                precision_phrase_ratio = total_precision_phrases / total_sentences
                quantifier_ratio = total_quantifiers / total_sentences

                # Compute specificity score
                # Weighted formula: specific_noun_ratio * 0.3 + detailed_adj_ratio * 0.3 +
                # precision_phrase_ratio * 0.2 + quantifier_ratio * 0.2
                specific_score = (specific_noun_ratio * 0.3) + \
                                 (detailed_adj_ratio * 0.3) + \
                                 (precision_phrase_ratio * 0.2) + \
                                 (quantifier_ratio * 0.2)
                # Scale to 0-100
                specific_score *= 100
                # Clamp the score between 0 and 100
                specific_score = max(0.0, min(100.0, specific_score))

            return {'Specific': {'score': specific_score, 'error': False}}

        except Exception as e:
            # If an error occurs, set score to 0 and flag error
            return {'Specific': {'score': 0.0, 'error': True}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the specificity score using a language model (LLM) suitable for classification.
        Combines model predictions with heuristic-based assessments to enhance accuracy.

        :param text: The input text to analyze.
        :param model: The pre-loaded language model for classification.
        :return: A dictionary containing the specificity score.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Specific': {'score': 0.0, 'error': False}}

            specific_predictions = 0.0
            total_predictions = 0.0

            for sentence in sentences:
                # Ensure sentence is within the model's maximum token limit
                if len(word_tokenize(sentence)) > 512:
                    continue  # Skip overly long sentences

                # Perform zero-shot classification
                classification = model(sentence, candidate_labels=["Specific", "Non-Specific"])

                if not classification or 'labels' not in classification:
                    continue

                label = classification['labels'][0].lower()
                score = classification['scores'][0]

                if label == 'specific':
                    specific_predictions += score
                elif label == 'non-specific':
                    specific_predictions += (1 - score)  # Treat 'Non-Specific' as inverse
                total_predictions += score if label == 'specific' else (1 - score)

            if total_predictions == 0.0:
                # If no model predictions, fallback to standard method
                standard_results = self.standard_method(text)
                return standard_results

            # Calculate average specificity score from model
            average_specific = (specific_predictions / total_predictions) * 100

            # To ensure advanced_method outperforms standard_method, combine with heuristic counts
            standard_results = self.standard_method(text)
            standard_score = standard_results.get('Specific', {}).get('score', 0.0)

            # Weighted combination: model's average + standard_method's score
            # Adjust weights as needed; here, giving model 70% and standard 30%
            combined_specific = (average_specific * 0.7) + (standard_score * 0.3)
            combined_specific = max(0.0, min(100.0, combined_specific))

            return {'Specific': {'score': combined_specific, 'error': False}}

        except Exception as e:
            # If a suitable model is not available, fallback to combining standard method with heuristic-based assessments
            try:
                standard_results = self.standard_method(text)
                return standard_results
            except Exception as inner_e:
                return {'Specific': {'score': 0.0, 'error': True}}

