# dimensions/Dynamic.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import Pipeline

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class Dynamic(BaseDimension):
    def __init__(self):
        super().__init__(name="Dynamic")

    def standard_method(self, text: str) -> dict:
        try:
            words = word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            total_words = len(words)
            # Count all verbs as dynamic words
            dynamic_verbs = [word for word, tag in pos_tags if tag.startswith('VB')]
            dynamic_score = (len(dynamic_verbs) / total_words) * 100 if total_words else 0.0
            return {'Dynamic': {'score': dynamic_score, 'error': False}}
        except Exception as e:
            return {'Dynamic': {'score': 0.0, 'error': True, 'error_message': str(e)}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return self.standard_method(text)

            dynamic_scores = []
            for sentence in sentences:
                result = model(sentence)
                if result:
                    label = result[0]['label'].lower()
                    if label in ['joy', 'surprise', 'excitement']:
                        dynamic_scores.append(result[0]['score'])

            if dynamic_scores:
                average_score = sum(dynamic_scores) / len(dynamic_scores) * 100
                return {'Dynamic': {'score': average_score, 'error': False}}
            else:
                return self.standard_method(text)
        except Exception as e:
            return self.standard_method(text)

'''
# dimensions/Dynamic.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, WordNetLemmatizer
from transformers.pipelines import Pipeline
import math  # Import math module

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

class Dynamic(BaseDimension):
    def __init__(self):
        super().__init__(name="Dynamic")
        self.dynamic_verbs = self.load_dynamic_verbs()

    def load_dynamic_verbs(self) -> set:
        """
        Loads a comprehensive set of dynamic verbs.
        """
        dynamic_verbs = set([
            # Expanded list of dynamic verbs
            "run", "jump", "swim", "dance", "drive", "fly", "build", "create",
            "move", "push", "pull", "lift", "throw", "catch", "kick", "hit",
            "eat", "drink", "write", "read", "sing", "paint", "play", "walk",
            "talk", "speak", "listen", "learn", "teach", "search", "explore",
            "travel", "race", "fight", "battle", "craft", "design", "develop",
            "program", "operate", "manage", "organize", "coordinate", "execute",
            "implement", "innovate", "solve", "analyze", "decide", "communicate",
            "control", "lead", "guide", "train", "assist", "support", "navigate",
            "repair", "maintain", "manufacture", "assemble", "construct", "demolish",
            "enhance", "improve", "upgrade", "modify", "adapt", "adjust", "transform",
            "revamp", "reorganize", "restructure", "optimize", "streamline", "simplify",
            "accelerate", "boost", "increase", "maximize", "expand", "extend", "grow",
            "scale", "augment", "heighten", "elevate", "rise", "soar", "surge",
            "escalate", "intensify", "amplify", "magnify", "change", "progress",
            "perform", "act", "initiate", "produce", "engage", "work", "study",
            "compete", "cook", "clean", "plan", "draw", "calculate", "measure",
            "investigate", "examine", "test", "experiment", "collaborate", "negotiate",
            "sail", "climb", "hike", "shoot", "score", "win", "lose", "attack",
            "defend", "conquer", "surrender", "invent", "discover", "launch",
            "update", "decelerate", "delegate", "motivate", "inspire", "advance",
            "pursue", "deploy", "orchestrate", "expedite", "propel", "revise",
            "pioneer", "venture", "evolve", "mobilize", "activate", "undertake",
            "facilitate", "generate", "foster", "stimulate", "incite", "ignite",
            "trigger", "induce", "instigate", "commence", "embark", "engender",
            "spur", "promote", "propagate", "plan", "organize", "drive", "change",
            "transform", "reduce", "implement", "develop"
        ])
        return dynamic_verbs

    def standard_method(self, text: str) -> dict:
        """
        Calculates the dynamic score based on the proportion of dynamic verbs.
        """
        try:
            lemmatizer = WordNetLemmatizer()
            words = word_tokenize(text)
            pos_tags = pos_tag(words)

            total_verbs = 0
            dynamic_count = 0

            for word, tag in pos_tags:
                if tag.startswith('VB'):
                    total_verbs += 1
                    lemma = lemmatizer.lemmatize(word, 'v').lower()
                    if lemma in self.dynamic_verbs:
                        dynamic_count += 1

            dynamic_score = (dynamic_count / total_verbs) * 100 if total_verbs else 0.0
            return {'Dynamic': {'score': dynamic_score, 'error': False}}
        except Exception as e:
            return {'Dynamic': {'score': 0.0, 'error': True, 'error_message': str(e)}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the dynamic score using sentiment variability.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Dynamic': {'score': 0.0, 'error': False}}

            sentiment_scores = []

            for sentence in sentences:
                predictions = model(sentence)

                if not predictions:
                    continue

                top_prediction = predictions[0]
                score = top_prediction['score']

                sentiment_scores.append(score)

            if not sentiment_scores:
                return {'Dynamic': {'score': 0.0, 'error': False}}

            # Calculate standard deviation of sentiment scores
            mean_score = sum(sentiment_scores) / len(sentiment_scores)
            variance = sum((s - mean_score) ** 2 for s in sentiment_scores) / len(sentiment_scores)
            std_dev = math.sqrt(variance)

            # Normalize standard deviation to a 0-100 scale
            normalized_std_dev = (std_dev / 0.5) * 100  # Assuming max std_dev ~0.5
            normalized_std_dev = max(0.0, min(100.0, normalized_std_dev))

            return {'Dynamic': {'score': normalized_std_dev, 'error': False}}

        except Exception as e:
            return {'Dynamic': {'score': 0.0, 'error': True, 'error_message': str(e)}}
'''



