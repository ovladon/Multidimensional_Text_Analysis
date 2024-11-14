# dimensions/Positive.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from transformers.pipelines import Pipeline

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

class Positive(BaseDimension):
    def __init__(self, method='standard'):
        super().__init__(name="Positive", method=method)
        # Initialize comprehensive sets of positive indicators
        self.positive_words = self.load_positive_words()
        self.positive_phrases = self.load_positive_phrases()
        self.affirmative_terms = self.load_affirmative_terms()
        self.positive_adjectives = self.load_positive_adjectives()
        # Compile regex patterns for performance
        self.positive_word_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in self.positive_words) + r')\b', 
            re.IGNORECASE
        )
        self.positive_phrase_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(phrase) for phrase in self.positive_phrases) + r')\b', 
            re.IGNORECASE
        )
        self.affirmative_term_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(term) for term in self.affirmative_terms) + r')\b', 
            re.IGNORECASE
        )
        self.positive_adj_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(adj) for adj in self.positive_adjectives) + r')\b', 
            re.IGNORECASE
        )

    def load_positive_words(self) -> set:
        """
        Loads a comprehensive set of positive-related words.
        
        :return: A set containing positive words.
        """
        positive_words = {
            "good", "great", "excellent", "fantastic", "amazing",
            "positive", "happy", "joyful", "delighted", "pleased",
            "content", "satisfied", "fortunate", "blessed", "thrilled",
            "elated", "cheerful", "uplifted", "optimistic", "hopeful",
            "encouraging", "beneficial", "favorable", "superb",
            "marvelous", "wonderful", "splendid", "terrific", "lovely",
            "charming", "graceful", "elegant", "brilliant", "radiant",
            "vibrant", "dynamic", "inspiring", "motivating", "rewarding",
            "enjoyable", "pleasurable", "exciting", "lively", "energetic",
            "invigorating", "refreshing", "stimulating", "colorful",
            "bright", "sparkling", "gleaming", "glowing", "shining",
            "dazzling", "sparkling", "resplendent", "glittering",
            "beaming", "twinkling", "bubbly", "spirited", "zestful"
        }
        return positive_words

    def load_positive_phrases(self) -> set:
        """
        Loads a comprehensive set of positive-related phrases.
        
        :return: A set containing positive phrases.
        """
        positive_phrases = {
            "in good spirits", "on cloud nine", "feeling great",
            "highly satisfied", "overjoyed", "full of joy",
            "bursting with happiness", "in high spirits", "in a good mood",
            "walking on air", "in seventh heaven", "thrilled to bits",
            "tickled pink", "grinning from ear to ear",
            "ecstatic about", "excited for", "looking forward to",
            "delighted with", "pleased to", "happy about",
            "content with", "satisfied with", "fortunate to",
            "blessed to", "joyful over", "cheerful about",
            "uplifted by", "optimistic about", "hopeful for",
            "encouraged by", "beneficial for", "favorable towards",
            "superb performance", "marvelous achievement",
            "wonderful experience", "splendid results", "terrific job",
            "lovely day", "charming personality", "graceful movement",
            "elegant design", "brilliant idea", "radiant smile",
            "vibrant colors", "dynamic environment", "inspiring leader",
            "motivating force", "rewarding experience", "enjoyable activity",
            "pleasurable moment", "exciting opportunity", "lively discussion",
            "energetic performance", "invigorating workout",
            "refreshing change", "stimulating conversation",
            "colorful display", "bright future", "sparkling wine",
            "gleaming surface", "glowing review", "shining example",
            "dazzling performance", "sparkling personality",
            "resplendent attire", "glittering lights", "beaming smile",
            "twinkling eyes", "bubbly personality", "spirited debate",
            "zestful approach"
        }
        return positive_phrases

    def load_affirmative_terms(self) -> set:
        """
        Loads a comprehensive set of affirmative terms that indicate positivity.
        
        :return: A set containing affirmative terms.
        """
        affirmative_terms = {
            "yes", "absolutely", "definitely", "certainly", "indeed",
            "sure", "of course", "without a doubt", "positively",
            "undoubtedly", "affirmative", "agree", "agreeing",
            "agreeable", "support", "supporting", "endorsing",
            "backing", "championing", "advocating", "encouraging",
            "validating", "confirming", "affirming", "ratifying",
            "approving", "sanctioning", "bolstering", "reinforcing"
        }
        return affirmative_terms

    def load_positive_adjectives(self) -> set:
        """
        Loads a comprehensive set of adjectives that inherently carry positive connotations.
        
        :return: A set containing positive adjectives.
        """
        positive_adjectives = {
            "amazing", "awesome", "beautiful", "breathtaking", "charming",
            "delightful", "enchanting", "fabulous", "fantastic", "glorious",
            "graceful", "incredible", "lovely", "magnificent", "marvelous",
            "pleasant", "radiant", "remarkable", "spectacular", "stunning",
            "superb", "terrific", "wonderful", "admirable", "alluring",
            "appealing", "attractive", "awesome", "brilliant", "captivating",
            "dazzling", "elegant", "exquisite", "fascinating", "glimmering",
            "gleaming", "impressive", "majestic", "mesmerizing", "pleasant",
            "polished", "prettiful", "refined", "resplendent", "sparkling",
            "splendid", "sumptuous", "vibrant", "vivacious", "winning",
            "winsome", "witty", "zesty", "zestful"
        }
        return positive_adjectives

    def standard_method(self, text: str) -> dict:
        """
        Calculates the positive score based on heuristic linguistic analysis.
    
        :param text: The input text to analyze.
        :return: A dictionary containing the positive score.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Positive': {'score': 0.0, 'error': False}}
    
            total_positive_words = 0
            total_positive_phrases = 0
            total_affirmative_terms = 0
            total_positive_adjectives = 0
            total_words = 0
            total_sentences = len(sentences)
    
            for sentence in sentences:
                words = word_tokenize(sentence)
                if not words:
                    continue
    
                total_words += len(words)
    
                # Count positive words
                positive_word_matches = self.positive_word_pattern.findall(sentence)
                total_positive_words += len(positive_word_matches)
    
                # Count positive phrases
                positive_phrase_matches = self.positive_phrase_pattern.findall(sentence)
                total_positive_phrases += len(positive_phrase_matches)
    
                # Count affirmative terms
                affirmative_term_matches = self.affirmative_term_pattern.findall(sentence)
                total_affirmative_terms += len(affirmative_term_matches)
    
                # Count positive adjectives
                positive_adj_matches = self.positive_adj_pattern.findall(sentence)
                total_positive_adjectives += len(positive_adj_matches)
    
            if total_words == 0:
                positive_score = 0.0
            else:
                # Calculate ratios
                positive_word_ratio = total_positive_words / total_words
                positive_phrase_ratio = total_positive_phrases / total_sentences
                affirmative_term_ratio = total_affirmative_terms / total_words
                positive_adj_ratio = total_positive_adjectives / total_words
    
                # Compute positive score
                # Weighted formula: positive_word_ratio * 0.3 + positive_phrase_ratio * 0.3 +
                # affirmative_term_ratio * 0.2 + positive_adj_ratio * 0.2
                positive_score = (positive_word_ratio * 0.3) + \
                                 (positive_phrase_ratio * 0.3) + \
                                 (affirmative_term_ratio * 0.2) + \
                                 (positive_adj_ratio * 0.2)
                # Scale to 0-100
                positive_score *= 100
                # Clamp the score between 0 and 100
                positive_score = max(0.0, min(100.0, positive_score))
    
            return {'Positive': {'score': positive_score, 'error': False}}
    
        except Exception as e:
            # If an error occurs, set score to 0 and flag error
            return {'Positive': {'score': 0.0, 'error': True}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the positive score using a language model (LLM) suitable for classification.
        Combines model predictions with heuristic-based assessments to enhance accuracy.
    
        :param text: The input text to analyze.
        :param model: The pre-loaded language model for classification.
        :return: A dictionary containing the positive score.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Positive': {'score': 0.0, 'error': False}}
    
            positive_predictions = 0.0
            total_predictions = 0.0
    
            for sentence in sentences:
                # Ensure sentence is within the model's maximum token limit
                if len(word_tokenize(sentence)) > 512:
                    continue  # Skip overly long sentences
    
                # Perform zero-shot classification
                classification = model(sentence, candidate_labels=["Positive", "Non-Positive"])
    
                if not classification or 'labels' not in classification:
                    continue
    
                label = classification['labels'][0].lower()
                score = classification['scores'][0]
    
                if label == 'positive':
                    positive_predictions += score
                elif label == 'non-positive':
                    positive_predictions += (1 - score)  # Treat 'Non-Positive' as inverse
                total_predictions += score if label == 'positive' else (1 - score)
    
            if total_predictions == 0.0:
                # If no model predictions, fallback to standard method
                standard_results = self.standard_method(text)
                return standard_results
    
            # Calculate average positive score from model
            average_positive = (positive_predictions / total_predictions) * 100
    
            # To ensure advanced_method outperforms standard_method, combine with heuristic counts
            standard_results = self.standard_method(text)
            standard_score = standard_results.get('Positive', {}).get('score', 0.0)
    
            # Weighted combination: model's average + standard_method's score
            # Adjust weights as needed; here, giving model 70% and standard 30%
            combined_positive = (average_positive * 0.7) + (standard_score * 0.3)
            combined_positive = max(0.0, min(100.0, combined_positive))
    
            return {'Positive': {'score': combined_positive, 'error': False}}
    
        except Exception as e:
            # If a suitable model is not available, fallback to combining standard method with heuristic-based assessments
            try:
                standard_results = self.standard_method(text)
                return standard_results
            except Exception as inner_e:
                return {'Positive': {'score': 0.0, 'error': True}}

