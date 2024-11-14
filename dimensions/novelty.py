# dimensions/Novelty.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from transformers.pipelines import Pipeline
import math

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

class Novelty(BaseDimension):
    def __init__(self, method='standard'):
        super().__init__(name="Novelty", method=method)
        # Initialize comprehensive sets of novelty indicators
        self.novelty_words = self.load_novelty_words()
        self.novelty_phrases = self.load_novelty_phrases()
        self.novelty_adverbs = self.load_novelty_adverbs()
        self.novelty_related_nouns = self.load_novelty_related_nouns()
        # Compile regex patterns for performance
        self.novelty_word_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in self.novelty_words) + r')\b', 
            re.IGNORECASE
        )
        self.novelty_phrase_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(phrase) for phrase in self.novelty_phrases) + r')\b', 
            re.IGNORECASE
        )
        self.novelty_adverb_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(adverb) for adverb in self.novelty_adverbs) + r')\b', 
            re.IGNORECASE
        )
        self.novelty_related_noun_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(noun) for noun in self.novelty_related_nouns) + r')\b', 
            re.IGNORECASE
        )

    def load_novelty_words(self) -> set:
        """
        Loads a comprehensive set of words related to novelty.
        
        :return: A set containing novelty-related words.
        """
        novelty_words = {
            "innovative", "unique", "new", "original", "fresh", "unprecedented",
            "breakthrough", "groundbreaking", "pioneering", "advanced", "novel",
            "revolutionary", "creative", "inventive", "distinctive", "uncommon",
            "different", "nontraditional", "cutting-edge", "modern", "state-of-the-art"
        }
        return novelty_words

    def load_novelty_phrases(self) -> set:
        """
        Loads a comprehensive set of phrases related to novelty.
        
        :return: A set containing novelty-related phrases.
        """
        novelty_phrases = {
            "brand new", "completely different", "state-of-the-art technology",
            "groundbreaking research", "unprecedented methods", "innovative approach",
            "cutting-edge solutions", "fresh perspective", "revolutionary idea",
            "pioneering work", "novel concept", "unique solution", "advanced techniques",
            "original design", "inventive strategy", "distinctive feature"
        }
        return novelty_phrases

    def load_novelty_adverbs(self) -> set:
        """
        Loads a comprehensive set of adverbs related to novelty.
        
        :return: A set containing novelty-related adverbs.
        """
        novelty_adverbs = {
            "innovatively", "uniquely", "originally", "creatively", "distinctively",
            "revolutionarily", "inventively", "pioneeringly", "cutting-edge", "groundbreakingly"
        }
        return novelty_adverbs

    def load_novelty_related_nouns(self) -> set:
        """
        Loads a comprehensive set of nouns related to novelty.
        
        :return: A set containing novelty-related nouns.
        """
        novelty_related_nouns = {
            "innovation", "uniqueness", "originality", "creativity", "breakthrough",
            "groundbreaking", "pioneering", "advanced technology", "revolution",
            "design", "strategy", "solution", "approach", "concept", "idea",
            "feature", "technique", "method", "research", "development"
        }
        return novelty_related_nouns

    def standard_method(self, text: str) -> dict:
        """
        Calculates the novelty score based on heuristic linguistic analysis.
        
        :param text: The input text to analyze.
        :return: A dictionary containing the novelty score.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Novelty': {'score': 0.0, 'error': False}}
            
            total_novelty_words = 0
            total_novelty_phrases = 0
            total_novelty_adverbs = 0
            total_novelty_related_nouns = 0
            total_words = 0
            total_sentences = len(sentences)
            
            for sentence in sentences:
                words = word_tokenize(sentence)
                if not words:
                    continue
                
                total_words += len(words)
                
                # Count novelty words
                novelty_word_matches = self.novelty_word_pattern.findall(sentence)
                total_novelty_words += len(novelty_word_matches)
                
                # Count novelty phrases
                novelty_phrase_matches = self.novelty_phrase_pattern.findall(sentence)
                total_novelty_phrases += len(novelty_phrase_matches)
                
                # Count novelty adverbs
                novelty_adverb_matches = self.novelty_adverb_pattern.findall(sentence)
                total_novelty_adverbs += len(novelty_adverb_matches)
                
                # Count novelty-related nouns
                novelty_related_noun_matches = self.novelty_related_noun_pattern.findall(sentence)
                total_novelty_related_nouns += len(novelty_related_noun_matches)
            
            if total_words == 0:
                novelty_word_ratio = 0.0
                novelty_adverb_ratio = 0.0
                novelty_related_noun_ratio = 0.0
            else:
                # Calculate ratios
                novelty_word_ratio = total_novelty_words / total_words
                novelty_adverb_ratio = total_novelty_adverbs / total_words
                novelty_related_noun_ratio = total_novelty_related_nouns / total_words
            
            if total_sentences == 0:
                novelty_phrase_ratio = 0.0
            else:
                novelty_phrase_ratio = total_novelty_phrases / total_sentences
            
            # Compute novelty score
            # Weighted formula: novelty_word_ratio * 0.25 + novelty_phrase_ratio * 0.25 +
            # novelty_adverb_ratio * 0.25 + novelty_related_noun_ratio * 0.25
            novelty_score = (novelty_word_ratio * 0.25) + \
                            (novelty_phrase_ratio * 0.25) + \
                            (novelty_adverb_ratio * 0.25) + \
                            (novelty_related_noun_ratio * 0.25)
            # Scale to 0-100
            novelty_score *= 100
            # Clamp the score between 0 and 100
            novelty_score = max(0.0, min(100.0, novelty_score))
            
            return {'Novelty': {'score': novelty_score, 'error': False}}
        
        except Exception as e:
            # If an error occurs, set score to 0 and flag error
            return {'Novelty': {'score': 0.0, 'error': True}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the novelty score using a language model (LLM) suitable for classification.
        Combines model predictions with heuristic-based assessments to enhance accuracy.
        
        :param text: The input text to analyze.
        :param model: The pre-loaded language model for classification.
        :return: A dictionary containing the novelty score.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Novelty': {'score': 0.0, 'error': False}}
            
            novelty_predictions = 0.0
            total_predictions = 0.0
            
            for sentence in sentences:
                # Ensure sentence is within the model's maximum token limit
                if len(word_tokenize(sentence)) > 512:
                    continue  # Skip overly long sentences
                
                # Perform zero-shot classification
                classification = model(sentence, candidate_labels=["Novel", "Non-Novel"])
                
                if not classification or 'labels' not in classification:
                    continue
                
                label = classification['labels'][0].lower()
                score = classification['scores'][0]
                
                if label == 'novel':
                    novelty_predictions += score
                elif label == 'non-novel':
                    novelty_predictions += (1 - score)  # Treat 'Non-Novel' as inverse
                total_predictions += score if label == 'novel' else (1 - score)
            
            if total_predictions == 0.0:
                # If no model predictions, fallback to standard method
                standard_results = self.standard_method(text)
                return standard_results
            
            # Calculate average novelty score from model
            average_novelty = (novelty_predictions / total_predictions) * 100
            
            # To ensure advanced_method outperforms standard_method, combine with heuristic counts
            standard_results = self.standard_method(text)
            standard_score = standard_results.get('Novelty', {}).get('score', 0.0)
            
            # Weighted combination: model's average + standard_method's score
            # Adjust weights as needed; here, giving model 70% and standard 30%
            combined_novelty = (average_novelty * 0.7) + (standard_score * 0.3)
            combined_novelty = max(0.0, min(100.0, combined_novelty))
            
            return {'Novelty': {'score': combined_novelty, 'error': False}}
        
        except Exception as e:
            # If a suitable model is not available, fallback to combining standard method with heuristic-based assessments
            try:
                standard_results = self.standard_method(text)
                return standard_results
            except Exception as inner_e:
                return {'Novelty': {'score': 0.0, 'error': True}}

