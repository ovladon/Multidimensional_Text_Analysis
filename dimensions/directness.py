# dimensions/Directness.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt', quiet=True)

class Directness(BaseDimension):
    def __init__(self):
        super().__init__(name="Directness")

    def standard_method(self, text: str) -> dict:
        try:
            sentences = sent_tokenize(text)
            total_sentences = len(sentences)
            direct_count = 0

            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                if words[0] in ["do", "make", "go", "stop", "start", "continue", "keep", "let", "help",
            "tell", "ask", "give", "take", "come", "leave", "begin", "finish",
            "listen", "look", "think", "try", "call", "put", "move", "change",
            "get", "find", "bring", "show", "follow", "wait", "remember", "pay",
            "write", "read", "believe", "turn", "hold", "work", "play",
            "run", "walk", "sit", "stand"]:
                    direct_count += 1
                elif any(word in ['could', 'would', 'might', 'may'] for word in words):
                    continue  # Indirect sentence
                else:
                    direct_count += 1  # Assume direct

            direct_score = (direct_count / total_sentences) * 100 if total_sentences else 0.0
            return {'Directness': {'score': direct_score, 'error': False}}
        except Exception as e:
            return {'Directness': {'score': 0.0, 'error': True, 'error_message': str(e)}}

    # Since appropriate models may not be available, we can rely on the standard method
    def advanced_method(self, text: str, model=None) -> dict:
        return self.standard_method(text)


'''
# dimensions/Directness.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import sent_tokenize
from transformers.pipelines import Pipeline  # Corrected import statement
import spacy

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

# Load spaCy model for dependency parsing
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')

class Directness(BaseDimension):
    def __init__(self):
        super().__init__(name="Directness")
        self.imperative_keywords = set([
            "do", "make", "go", "stop", "start", "continue", "keep", "let", "help",
            "tell", "ask", "give", "take", "come", "leave", "begin", "finish",
            "listen", "look", "think", "try", "call", "put", "move", "change",
            "get", "find", "bring", "show", "follow", "wait", "remember", "pay",
            "write", "read", "believe", "turn", "hold", "work", "play",
            "run", "walk", "sit", "stand"
        ])

    def standard_method(self, text: str) -> dict:
        """
        Calculates the directness score using dependency parsing with spaCy.
        """
        try:
            doc = nlp(text)
            direct_count = 0
            total_sentences = len(list(doc.sents))

            if total_sentences == 0:
                return {'Directness': {'score': 0.0, 'error': False}}

            for sent in doc.sents:
                tokens = [token for token in sent]
                if not tokens:
                    continue

                # Check for imperative mood
                if tokens[0].pos_ == 'VERB' and tokens[0].dep_ == 'ROOT':
                    direct_count += 1
                # Check for first-person singular/plural subjects
                elif any(token.text.lower() in ['i', 'we'] and token.dep_ == 'nsubj' for token in tokens):
                    direct_count += 1
                # Check for imperative keywords
                elif any(token.lemma_.lower() in self.imperative_keywords for token in tokens if token.pos_ == 'VERB'):
                    direct_count += 1

            directness_score = (direct_count / total_sentences) * 100
            return {'Directness': {'score': directness_score, 'error': False}}
        except Exception as e:
            return {'Directness': {'score': 0.0, 'error': True, 'error_message': str(e)}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the directness score using a text classification model.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Directness': {'score': 0.0, 'error': False}}

            direct_predictions = 0.0
            total_predictions = 0

            for sentence in sentences:
                predictions = model(sentence)

                if not predictions:
                    continue

                top_prediction = predictions[0]
                label = top_prediction['label'].lower()
                score = top_prediction['score']

                # Map model labels to 'direct' and 'indirect'
                if label in ['direct', 'LABEL_1']:
                    direct_predictions += score
                elif label in ['indirect', 'LABEL_0']:
                    direct_predictions += (1 - score)
                else:
                    # If label is unrecognized, skip this sentence
                    continue

                total_predictions += 1

            if total_predictions == 0:
                return {'Directness': {'score': 0.0, 'error': False}}

            directness_score = (direct_predictions / total_predictions) * 100
            return {'Directness': {'score': directness_score, 'error': False}}
        except Exception as e:
            return {'Directness': {'score': 0.0, 'error': True, 'error_message': str(e)}}
'''

