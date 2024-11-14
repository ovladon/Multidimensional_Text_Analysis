# dimensions/generic.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import Pipeline

nltk.download('punkt', quiet=True)

class Generic(BaseDimension):
    def __init__(self):
        super().__init__(name="Generic")

    def standard_method(self, text: str) -> dict:
        try:
            words = word_tokenize(text.lower())
            total_words = len(words)
            generic_words = set(['thing', 'stuff', 'object', 'items', 'goods', 'entities', 'elements', 'aspects'])
            generic_count = sum(1 for word in words if word in generic_words)
            generic_score = (generic_count / total_words) * 100 if total_words else 0.0
            return {'Generic': {'score': generic_score, 'error': False}}
        except Exception as e:
            return {'Generic': {'score': 0.0, 'error': True, 'error_message': str(e)}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return self.standard_method(text)

            generic_scores = []
            for sentence in sentences:
                result = model(sentence)
                if result:
                    label = result[0]['label'].lower()
                    score = result[0]['score']
                    if 'generic' in label:
                        generic_scores.append(score)
                    elif 'specific' in label:
                        generic_scores.append(1 - score)

            if generic_scores:
                average_score = sum(generic_scores) / len(generic_scores) * 100
                return {'Generic': {'score': average_score, 'error': False}}
            else:
                return self.standard_method(text)
        except Exception as e:
            return self.standard_method(text)


'''
# dimensions/generic.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize
from transformers.pipelines import Pipeline
import re

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

class Generic(BaseDimension):
    def __init__(self):
        super().__init__(name="Generic")
        self.generic_keywords = self.load_generic_keywords()
        self.specific_keywords = self.load_specific_keywords()
        # Compile regex patterns
        self.generic_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, self.generic_keywords)) + r')\b', re.IGNORECASE)
        self.specific_pattern = re.compile('|'.join(map(re.escape, self.specific_keywords)), re.IGNORECASE)

    def load_generic_keywords(self) -> set:
        """
        Loads a comprehensive set of generic words.
        """
        generic_words = set([
            # Expanded list of generic terms
            "people", "things", "stuff", "items", "objects", "entities", "ideas",
            "concepts", "elements", "aspects", "factors", "components", "areas",
            "fields", "issues", "topics", "matters", "subjects", "affairs",
            "events", "activities", "processes", "procedures", "methods", "systems",
            "structures", "mechanisms", "devices", "tools", "means", "ways",
            "approaches", "strategies", "techniques", "tactics", "solutions",
            "options", "alternatives", "possibilities", "opportunities", "advantages",
            "benefits", "challenges", "problems", "difficulties", "risks",
            "threats", "questions", "answers", "responses", "outcomes", "results",
            "effects", "impacts", "influences", "changes", "developments",
            "trends", "patterns", "movements", "directions", "destinations",
            "locations", "positions", "situations", "conditions", "states",
            "levels", "degrees", "forms", "types", "kinds", "categories",
            "classes", "groups", "sets", "series", "ranges", "varieties",
            "species", "genera", "families", "orders", "kingdoms"
        ])
        return generic_words

    def load_specific_keywords(self) -> set:
        """
        Loads a comprehensive set of specific words.
        """
        specific_words = set([
            # Expanded list of specific terms
            "quantum physics", "genetic engineering", "artificial intelligence",
            "neural networks", "climate change", "global warming", "carbon dioxide",
            "photosynthesis", "cellular respiration", "plate tectonics",
            "black holes", "supernova", "DNA sequencing", "RNA transcription",
            "protein synthesis", "nanotechnology", "cryptocurrency", "blockchain",
            "internet of things", "augmented reality", "virtual reality",
            "machine learning", "deep learning", "reinforcement learning",
            "natural language processing", "computer vision", "3D printing",
            "genome editing", "CRISPR", "stem cells", "immunotherapy",
            "renewable energy", "solar power", "wind energy", "geothermal energy",
            "hydroelectric power", "biomass energy", "quantum computing",
            "fiber optics", "semiconductors", "microprocessors", "superconductors",
            "robotics", "drone technology", "autonomous vehicles", "self-driving cars",
            "electric vehicles", "space exploration", "Mars rover", "satellite imaging",
            "weather forecasting", "oceanography", "meteorology", "astrophysics",
            "biochemistry", "molecular biology", "ecosystems", "biodiversity",
            "conservation biology", "environmental science", "anthropology",
            "sociology", "psychology", "neuroscience", "linguistics",
            "philosophy", "theology", "archaeology", "history", "literature",
            "poetry", "music theory", "art history", "film studies",
            "gender studies", "cultural studies", "political science", "economics",
            "finance", "marketing", "management", "accounting", "law",
            "medicine", "nursing", "pharmacy", "dentistry", "veterinary medicine",
            "engineering", "civil engineering", "mechanical engineering",
            "electrical engineering", "chemical engineering", "software engineering",
            "computer science", "information technology", "cybersecurity"
        ])
        return specific_words

    def standard_method(self, text: str) -> dict:
        """
        Calculates the genericity score based on heuristic linguistic analysis.
        """
        try:
            total_words = len(word_tokenize(text))
            if total_words == 0:
                return {'Generic': {'score': 0.0, 'error': False}}

            generic_matches = self.generic_pattern.findall(text)
            specific_matches = self.specific_pattern.findall(text)

            generic_count = len(generic_matches)
            specific_count = len(specific_matches)

            # Calculate score, ensure it's not negative
            generic_score = max(0.0, ((generic_count - specific_count) / total_words) * 100)

            return {'Generic': {'score': generic_score, 'error': False}}
        except Exception as e:
            return {'Generic': {'score': 0.0, 'error': True, 'error_message': str(e)}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the genericity score using a text classification model.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Generic': {'score': 0.0, 'error': False}}

            generic_predictions = 0.0
            total_predictions = 0.0

            for sentence in sentences:
                predictions = model(sentence)

                if not predictions:
                    continue

                top_prediction = predictions[0]
                label = top_prediction['label'].lower()
                score = top_prediction['score']

                if 'generic' in label:
                    generic_predictions += score
                elif 'specific' in label:
                    generic_predictions += (1 - score)
                total_predictions += 1

            if total_predictions == 0.0:
                return {'Generic': {'score': 0.0, 'error': False}}

            average_genericity = (generic_predictions / total_predictions) * 100
            average_genericity = max(0.0, min(100.0, average_genericity))

            return {'Generic': {'score': average_genericity, 'error': False}}

        except Exception as e:
            return {'Generic': {'score': 0.0, 'error': True, 'error_message': str(e)}}
'''

