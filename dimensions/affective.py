# dimensions/Affective.py

from .base_dimension import BaseDimension
import nltk
from nltk.corpus import opinion_lexicon
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers.pipelines import Pipeline

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('opinion_lexicon', quiet=True)

class Affective(BaseDimension):
    def __init__(self, method='standard'):
        super().__init__(name="Affective", method=method)

    def standard_method(self, text):
        """
        Calculates affective score using NLTK's opinion lexicon.
        Detects positive and negative emotions in the text.
        
        :param text: The input text to analyze.
        :return: A dictionary containing the affective score.
        """
        # Load positive and negative words from opinion lexicon
        positive_words = set(opinion_lexicon.positive())
        negative_words = set(opinion_lexicon.negative())

        # Tokenize text into words
        tokens = word_tokenize(text.lower())

        # Count positive and negative words
        pos_count = sum(1 for word in tokens if word in positive_words)
        neg_count = sum(1 for word in tokens if word in negative_words)

        # Calculate affective score as (pos - neg) / total words * 100
        total_words = len(tokens)
        if total_words == 0:
            return {'Affective': {'score': 0.0, 'error': False}}
        affective_score = ((pos_count - neg_count) / total_words) * 100
        # Clamp the score between 0 and 100
        affective_score = max(0.0, min(100.0, affective_score))
        return {'Affective': {'score': affective_score, 'error': False}}

    def advanced_method(self, text, model):
        """
        Calculates affective score using sentiment analysis via the provided model.
        
        :param text: The input text to analyze.
        :param model: The pre-loaded sentiment analysis model.
        :return: A dictionary containing the affective score.
        """
        try:
            # Tokenize the text into sentences
            sentences = sent_tokenize(text)
            total_score = 0
            num_sentences = len(sentences)

            if num_sentences == 0:
                return {'Affective': {'score': 0.0, 'error': False}}

            # Analyze each sentence and accumulate scores
            for sentence in sentences:
                # Truncate sentences to 512 characters if needed
                truncated_sentence = sentence[:512]
                result = model(truncated_sentence)[0]  # Assuming the model returns a list of dicts
                label = result['label']
                # Convert labels to numerical scores
                if 'star' in label.lower():
                    # e.g., '3 stars'
                    stars = int(label.split()[0])
                    total_score += stars
                elif 'positive' in label.lower():
                    total_score += 5  # Assign higher scores for positive sentiments
                elif 'negative' in label.lower():
                    total_score += 1  # Assign lower scores for negative sentiments
                else:
                    total_score += 3  # Assign neutral score

            # Calculate average score and scale to 0-100
            average_score = (total_score / (num_sentences * 5)) * 100  # Assuming max score per sentence is 5
            # Clamp the score between 0 and 100
            average_score = max(0.0, min(100.0, average_score))
            return {'Affective': {'score': average_score, 'error': False}}

        except Exception as e:
            # If an error occurs, set score to 0 and flag error
            return {'Affective': {'score': 0.0, 'error': True}}

