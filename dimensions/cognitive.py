# dimensions/Cognitive.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from transformers.pipelines import Pipeline

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

class Cognitive(BaseDimension):
    def __init__(self, method='standard'):
        super().__init__(name="Cognitive", method=method)
        # Initialize comprehensive sets of cognitive indicators
        self.cognitive_words = self.load_cognitive_words()
        self.cognitive_phrases = self.load_cognitive_phrases()
        self.mental_process_terms = self.load_mental_process_terms()
        self.intellectual_adjectives = self.load_intellectual_adjectives()
        # Compile regex patterns for performance
        self.cognitive_word_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in self.cognitive_words) + r')\b', 
            re.IGNORECASE
        )
        self.cognitive_phrase_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(phrase) for phrase in self.cognitive_phrases) + r')\b', 
            re.IGNORECASE
        )
        self.mental_process_term_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(term) for term in self.mental_process_terms) + r')\b', 
            re.IGNORECASE
        )
        self.intellectual_adj_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(adj) for adj in self.intellectual_adjectives) + r')\b', 
            re.IGNORECASE
        )

    def load_cognitive_words(self) -> set:
        """
        Loads a comprehensive set of cognitive-related words.
        
        :return: A set containing cognitive words.
        """
        cognitive_words = {
            "think", "thought", "mind", "cognition", "reasoning",
            "perception", "memory", "attention", "awareness",
            "intelligence", "logic", "understanding", "comprehension",
            "insight", "analysis", "synthesis", "evaluation",
            "problem-solving", "decision-making", "creativity",
            "imagination", "concentration", "reflection", "concept",
            "idea", "belief", "opinion", "judgment", "perspective",
            "awareness", "mental", "cognitive", "intellectual"
        }
        return cognitive_words

    def load_cognitive_phrases(self) -> set:
        """
        Loads a comprehensive set of cognitive-related phrases.
        
        :return: A set containing cognitive phrases.
        """
        cognitive_phrases = {
            "critical thinking", "problem solving", "logical reasoning",
            "cognitive process", "mental ability", "intellectual capacity",
            "cognitive function", "memory recall", "attention span",
            "decision making", "creative thinking", "analytical skills",
            "cognitive development", "mental flexibility", "conceptual understanding",
            "information processing", "strategic thinking", "abstract reasoning",
            "cognitive load", "cognitive bias", "cognitive dissonance",
            "cognitive framework", "cognitive architecture", "cognitive neuroscience"
        }
        return cognitive_phrases

    def load_mental_process_terms(self) -> set:
        """
        Loads a comprehensive set of mental process-related terms.
        
        :return: A set containing mental process terms.
        """
        mental_process_terms = {
            "analyze", "synthesize", "evaluate", "assess", "interpret",
            "compare", "contrast", "deduce", "infer", "predict",
            "hypothesize", "organize", "classify", "sequence",
            "abstract", "generalize", "specificize", "model",
            "simulate", "strategize", "decide", "solve", "create"
        }
        return mental_process_terms

    def load_intellectual_adjectives(self) -> set:
        """
        Loads a comprehensive set of adjectives that indicate intellectual depth.
        
        :return: A set containing intellectual adjectives.
        """
        intellectual_adjectives = {
            "analytical", "critical", "logical", "systematic",
            "reflective", "thoughtful", "inquisitive", "curious",
            "insightful", "innovative", "creative", "strategic",
            "abstract", "conceptual", "methodical", "disciplined",
            "evaluative", "sophisticated", "meticulous", "precise",
            "reasoned", "rational", "well-informed", "educated",
            "knowledgeable", "erudite", "intelligent", "wise"
        }
        return intellectual_adjectives

    def standard_method(self, text: str) -> dict:
        """
        Calculates the cognitive score based on heuristic linguistic analysis.
    
        :param text: The input text to analyze.
        :return: A dictionary containing the cognitive score.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Cognitive': {'score': 0.0, 'error': False}}
    
            total_cognitive_words = 0
            total_cognitive_phrases = 0
            total_mental_process_terms = 0
            total_intellectual_adjectives = 0
            total_words = 0
            total_sentences = len(sentences)
    
            for sentence in sentences:
                words = word_tokenize(sentence)
                if not words:
                    continue
    
                total_words += len(words)
    
                # Count cognitive words
                cognitive_word_matches = self.cognitive_word_pattern.findall(sentence)
                total_cognitive_words += len(cognitive_word_matches)
    
                # Count cognitive phrases
                cognitive_phrase_matches = self.cognitive_phrase_pattern.findall(sentence)
                total_cognitive_phrases += len(cognitive_phrase_matches)
    
                # Count mental process terms
                mental_process_term_matches = self.mental_process_term_pattern.findall(sentence)
                total_mental_process_terms += len(mental_process_term_matches)
    
                # Count intellectual adjectives
                intellectual_adj_matches = self.intellectual_adj_pattern.findall(sentence)
                total_intellectual_adjectives += len(intellectual_adj_matches)
    
            if total_words == 0:
                cognitive_score = 0.0
            else:
                # Calculate ratios
                cognitive_word_ratio = total_cognitive_words / total_words
                cognitive_phrase_ratio = total_cognitive_phrases / total_sentences
                mental_process_term_ratio = total_mental_process_terms / total_words
                intellectual_adj_ratio = total_intellectual_adjectives / total_words
    
                # Compute cognitive score
                # Weighted formula: cognitive_word_ratio * 0.3 + cognitive_phrase_ratio * 0.3 +
                # mental_process_term_ratio * 0.2 + intellectual_adj_ratio * 0.2
                cognitive_score = (cognitive_word_ratio * 0.3) + \
                                  (cognitive_phrase_ratio * 0.3) + \
                                  (mental_process_term_ratio * 0.2) + \
                                  (intellectual_adj_ratio * 0.2)
                # Scale to 0-100
                cognitive_score *= 100
                # Clamp the score between 0 and 100
                cognitive_score = max(0.0, min(100.0, cognitive_score))
    
            return {'Cognitive': {'score': cognitive_score, 'error': False}}
    
        except Exception as e:
            # If an error occurs, set score to 0 and flag error
            return {'Cognitive': {'score': 0.0, 'error': True}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the cognitive score using a language model (LLM) suitable for classification.
        Combines model predictions with heuristic-based assessments to enhance accuracy.
    
        :param text: The input text to analyze.
        :param model: The pre-loaded language model for classification.
        :return: A dictionary containing the cognitive score.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Cognitive': {'score': 0.0, 'error': False}}
    
            cognitive_predictions = 0.0
            total_predictions = 0.0
    
            for sentence in sentences:
                # Ensure sentence is within the model's maximum token limit
                if len(word_tokenize(sentence)) > 512:
                    continue  # Skip overly long sentences
    
                # Perform zero-shot classification
                classification = model(sentence, candidate_labels=["Cognitive", "Non-Cognitive"])
    
                if not classification or 'labels' not in classification:
                    continue
    
                label = classification['labels'][0].lower()
                score = classification['scores'][0]
    
                if label == 'cognitive':
                    cognitive_predictions += score
                elif label == 'non-cognitive':
                    cognitive_predictions += (1 - score)  # Treat 'Non-Cognitive' as inverse
                total_predictions += score if label == 'cognitive' else (1 - score)
    
            if total_predictions == 0.0:
                # If no model predictions, fallback to standard method
                standard_results = self.standard_method(text)
                return standard_results
    
            # Calculate average cognitive score from model
            average_cognitive = (cognitive_predictions / total_predictions) * 100
    
            # To ensure advanced_method outperforms standard_method, combine with heuristic counts
            standard_results = self.standard_method(text)
            standard_score = standard_results.get('Cognitive', {}).get('score', 0.0)
    
            # Weighted combination: model's average + standard_method's score
            # Adjust weights as needed; here, giving model 70% and standard 30%
            combined_cognitive = (average_cognitive * 0.7) + (standard_score * 0.3)
            combined_cognitive = max(0.0, min(100.0, combined_cognitive))
    
            return {'Cognitive': {'score': combined_cognitive, 'error': False}}
    
        except Exception as e:
            # If a suitable model is not available, fallback to combining standard method with heuristic-based assessments
            try:
                standard_results = self.standard_method(text)
                return standard_results
            except Exception as inner_e:
                return {'Cognitive': {'score': 0.0, 'error': True}}

