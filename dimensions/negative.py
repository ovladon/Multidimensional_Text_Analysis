# dimensions/Negative.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from transformers.pipelines import Pipeline
import math

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

class Negative(BaseDimension):
    def __init__(self, method='standard'):
        super().__init__(name="Negative", method=method)
        # Initialize comprehensive sets of negative indicators
        self.negative_words = self.load_negative_words()
        self.negative_phrases = self.load_negative_phrases()
        self.negation_terms = self.load_negation_terms()
        self.adjective_negatives = self.load_adjective_negatives()
        # Compile regex patterns for performance
        self.negative_word_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in self.negative_words) + r')\b', 
            re.IGNORECASE
        )
        self.negative_phrase_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(phrase) for phrase in self.negative_phrases) + r')\b', 
            re.IGNORECASE
        )
        self.negation_term_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(term) for term in self.negation_terms) + r')\b', 
            re.IGNORECASE
        )
        self.adjective_negative_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(adj) for adj in self.adjective_negatives) + r')\b', 
            re.IGNORECASE
        )

    def load_negative_words(self) -> set:
        """
        Loads a comprehensive set of negative-related words.
        
        :return: A set containing negative words.
        """
        negative_words = {
            "bad", "worse", "worst", "poor", "negative", "hate", "dislike",
            "annoy", "frustrate", "upset", "angry", "furious", "irritate",
            "disgust", "depress", "dishearten", "dismay", "sadden",
            "embarrass", "confuse", "shock", "panic", "fear", "scare",
            "terrify", "horrify", "alarm", "intimidate", "jeopardize",
            "harm", "injure", "damage", "destroy", "ruin", "wreck",
            "wreckage", "cripple", "disable", "impair", "undermine",
            "weaken", "betray", "cheat", "deceive", "mislead", "betray",
            "dishonest", "unreliable", "untrustworthy", "corrupt",
            "greedy", "selfish", "mean", "cruel", "unkind", "harsh",
            "brutal", "vicious", "malevolent", "malicious", "spiteful",
            "vindictive", "ruthless", "heartless", "callous", "cold",
            "aloof", "detached", "indifferent", "apathetic", "unfeeling",
            "unconcerned", "ignorant", "incompetent", "inept", "clumsy",
            "ineffective", "inefficient", "inadequate", "lacking",
            "insufficient", "deficient", "subpar", "inferior", "mediocre",
            "unsatisfactory", "unacceptable", "disappointing", "unpleasant",
            "unhappy", "sorrowful", "miserable", "dejected", "despondent",
            "forlorn", "gloomy", "bleak", "dismal", "morose", "somber",
            "melancholic", "doleful", "woeful", "sullen", "glum"
        }
        return negative_words

    def load_negative_phrases(self) -> set:
        """
        Loads a comprehensive set of negative-related phrases.
        
        :return: A set containing negative phrases.
        """
        negative_phrases = {
            "not good", "not happy", "not pleased", "not satisfied",
            "very bad", "extremely poor", "highly negative",
            "absolutely terrible", "completely awful", "utterly disappointing",
            "totally unacceptable", "deeply troubled", "seriously concerned",
            "strongly opposed", "heavily criticized", "firmly believe",
            "hardly ever", "barely any", "rarely seen", "seldom heard",
            "lack of", "absence of", "no improvement", "no progress",
            "no change", "failing to", "unable to", "cannot handle",
            "struggling with", "difficulty in", "challenge in",
            "problem with", "issue in", "complication with", "hassle with",
            "grievance about", "complaint regarding", "objection to",
            "disapproval of", "resentment towards", "discontent with",
            "frustration over", "exasperation with", "annoyance at",
            "irritation with", "agitation about", "distress over",
            "panic due to", "fear of", "terror from", "horror at",
            "apprehension about", "anxiety over", "nervous about",
            "unease regarding", "disquiet about", "restlessness due to"
        }
        return negative_phrases

    def load_negation_terms(self) -> set:
        """
        Loads a comprehensive set of negation terms that indicate the absence of positivity.
        
        :return: A set containing negation terms.
        """
        negation_terms = {
            "not", "never", "no", "none", "nothing", "nowhere", "hardly",
            "scarcely", "rarely", "seldom", "without", "lack of",
            "inability to", "fail to", "cannot", "can't", "won't",
            "doesn't", "don't", "didn't", "isn't", "aren't", "wasn't",
            "weren't", "haven't", "hasn't", "hadn't", "won't", "wouldn't",
            "shouldn't", "couldn't", "mightn't", "mustn't"
        }
        return negation_terms

    def load_adjective_negatives(self) -> set:
        """
        Loads a comprehensive set of adjectives that inherently carry negative connotations.
        
        :return: A set containing negative adjectives.
        """
        adjective_negatives = {
            "horrible", "awful", "dreadful", "appalling", "atrocious",
            "abysmal", "dire", "grim", "ghastly", "hideous",
            "loathsome", "repulsive", "vile", "foul", "nasty",
            "obnoxious", "offensive", "distasteful", "gross",
            "repellent", "hateful", "abominable", "heinous",
            "despicable", "contemptible", "scandalous", "shameful",
            "deplorable", "disgraceful", "unacceptable", "unbearable",
            "unfortunate", "unpleasant", "undesirable", "undeserved",
            "unwarranted", "inappropriate", "improper", "unethical",
            "unsuitable", "inadequate", "inferior", "deficient",
            "defective", "faulty", "flawed", "substandard",
            "mediocre", "ineffective", "inefficient", "inept",
            "clumsy", "bungling", "awkward", "blundering",
            "unskilled", "untrained", "unqualified", "unfit",
            "incompetent", "inelegant", "unrefined", "vulgar",
            "crude", "rough", "tawdry", "garish", "flashy",
            "gaudy", "tacky", "sleazy", "shabby", "unkempt",
            "unkind", "harsh", "cruel", "merciless", "ruthless",
            "malevolent", "spiteful", "vindictive", "vengeful",
            "hateful", "hostile", "antagonistic", "belligerent",
            "aggressive", "abrasive", "abrasive", "cutting",
            "caustic", "acerbic", "biting", "sarcastic", "snarky",
            "snide", "mordant", "trenchant", "scathing",
            "savage", "fierce", "ferocious", "brutal", "violent",
            "savage", "torturous", "grievous", "agony-inducing",
            "painful", "agonizing", "excruciating", "unrelenting",
            "relentless", "merciless", "torturous"
        }
        return adjective_negatives

    def standard_method(self, text: str) -> dict:
        """
        Calculates the negative score based on heuristic linguistic analysis.
    
        :param text: The input text to analyze.
        :return: A dictionary containing the negative score.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Negative': {'score': 0.0, 'error': False}}
    
            total_negative_words = 0
            total_negative_phrases = 0
            total_negation_terms = 0
            total_adjective_negatives = 0
            total_words = 0
            total_sentences = len(sentences)
    
            for sentence in sentences:
                words = word_tokenize(sentence)
                if not words:
                    continue
    
                total_words += len(words)
    
                # Count negative words
                negative_word_matches = self.negative_word_pattern.findall(sentence)
                total_negative_words += len(negative_word_matches)
    
                # Count negative phrases
                negative_phrase_matches = self.negative_phrase_pattern.findall(sentence)
                total_negative_phrases += len(negative_phrase_matches)
    
                # Count negation terms
                negation_term_matches = self.negation_term_pattern.findall(sentence)
                total_negation_terms += len(negation_term_matches)
    
                # Count negative adjectives
                adjective_negative_matches = self.adjective_negative_pattern.findall(sentence)
                total_adjective_negatives += len(adjective_negative_matches)
    
            if total_words == 0:
                negative_score = 0.0
            else:
                # Calculate ratios
                negative_word_ratio = total_negative_words / total_words
                negative_phrase_ratio = total_negative_phrases / total_sentences
                negation_term_ratio = total_negation_terms / total_words
                adjective_negative_ratio = total_adjective_negatives / total_words
    
                # Compute negative score
                # Weighted formula: negative_word_ratio * 0.3 + negative_phrase_ratio * 0.3 +
                # negation_term_ratio * 0.2 + adjective_negative_ratio * 0.2
                negative_score = (negative_word_ratio * 0.3) + \
                                 (negative_phrase_ratio * 0.3) + \
                                 (negation_term_ratio * 0.2) + \
                                 (adjective_negative_ratio * 0.2)
                # Scale to 0-100
                negative_score *= 100
                # Clamp the score between 0 and 100
                negative_score = max(0.0, min(100.0, negative_score))
    
            return {'Negative': {'score': negative_score, 'error': False}}
    
        except Exception as e:
            # If an error occurs, set score to 0 and flag error
            return {'Negative': {'score': 0.0, 'error': True}}
    
    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the negative score using a language model (LLM) suitable for classification.
        Combines model predictions with heuristic-based assessments to enhance accuracy.
    
        :param text: The input text to analyze.
        :param model: The pre-loaded language model for classification.
        :return: A dictionary containing the negative score.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Negative': {'score': 0.0, 'error': False}}
    
            negative_predictions = 0.0
            total_predictions = 0.0
    
            for sentence in sentences:
                # Ensure sentence is within the model's maximum token limit
                if len(word_tokenize(sentence)) > 512:
                    continue  # Skip overly long sentences
    
                # Perform zero-shot classification
                classification = model(sentence, candidate_labels=["Negative", "Non-Negative"])
    
                if not classification or 'labels' not in classification:
                    continue
    
                label = classification['labels'][0].lower()
                score = classification['scores'][0]
    
                if label == 'negative':
                    negative_predictions += score
                elif label == 'non-negative':
                    negative_predictions += (1 - score)  # Treat 'Non-Negative' as inverse
                total_predictions += score if label == 'negative' else (1 - score)
    
            if total_predictions == 0.0:
                # If no model predictions, fallback to standard method
                standard_results = self.standard_method(text)
                return standard_results
    
            # Calculate average negative score from model
            average_negative = (negative_predictions / total_predictions) * 100
    
            # To ensure advanced_method outperforms standard_method, combine with heuristic counts
            standard_results = self.standard_method(text)
            standard_score = standard_results.get('Negative', {}).get('score', 0.0)
    
            # Weighted combination: model's average + standard_method's score
            # Adjust weights as needed; here, giving model 70% and standard 30%
            combined_negative = (average_negative * 0.7) + (standard_score * 0.3)
            combined_negative = max(0.0, min(100.0, combined_negative))
    
            return {'Negative': {'score': combined_negative, 'error': False}}
    
        except Exception as e:
            # If a suitable model is not available, fallback to combining standard method with heuristic-based assessments
            try:
                standard_results = self.standard_method(text)
                return standard_results
            except Exception as inner_e:
                return {'Negative': {'score': 0.0, 'error': True}}

