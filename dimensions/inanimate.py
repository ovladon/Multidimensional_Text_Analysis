# dimensions/Inanimate.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

class Inanimate(BaseDimension):
    def __init__(self):
        super().__init__(name="Inanimate")
        self.inanimate_words = set(["city", "technology", "idea", "concept", "building", "road", "vehicle",
            "computer", "phone", "book", "instrument", "tool", "device", "machine",
            "equipment", "furniture", "material", "substance", "chemical", "metal",
            "plastic", "glass", "fabric", "paper", "water", "air", "fire", "earth",
            "stone", "rock", "mountain", "river", "ocean", "lake", "forest", "desert",
            "planet", "star", "galaxy", "universe", "atom", "molecule", "cell",
            "organ", "tissue", "system", "network", "structure", "algorithm",
            "software", "hardware", "data", "information", "code", "signal",
            "energy", "force", "power", "light", "sound", "color", "shape",
            "size", "temperature", "pressure", "gravity", "magnetism", "electricity",
            "currency", "money", "economy", "market", "industry", "product",
            "service", "policy", "law", "rule", "regulation", "contract",
            "agreement", "document", "report", "plan", "strategy", "project",
            "task", "goal", "objective", "result", "outcome", "effect",
            "impact", "change", "development", "trend", "pattern", "system",
            "method", "approach", "process", "procedure", "practice", "technique",
            "style", "design", "model", "framework", "theory", "hypothesis",
            "experiment", "test", "analysis", "evaluation", "measurement",
            "assessment", "review", "summary", "overview", "description",
            "explanation", "definition", "concept", "idea", "notion", "thought",
            "belief", "value", "principle", "standard", "norm", "criterion",
            "parameter", "variable", "function", "operation", "calculation",
            "equation", "formula", "algorithm", "model", "simulation", "representation"])

    def standard_method(self, text: str) -> dict:
        try:
            words = word_tokenize(text.lower())
            total_words = len(words)
            inanimate_count = sum(1 for word in words if word in self.inanimate_words)
            inanimate_score = (inanimate_count / total_words) * 100 if total_words else 0.0
            return {'Inanimate': {'score': inanimate_score, 'error': False}}
        except Exception as e:
            return {'Inanimate': {'score': 0.0, 'error': True, 'error_message': str(e)}}


'''
# dimensions/inanimate.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from itertools import chain  # Import chain
from transformers.pipelines import Pipeline

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class Inanimate(BaseDimension):
    def __init__(self):
        super().__init__(name="Inanimate")
        self.inanimate_nouns = self.load_inanimate_nouns()

    def load_inanimate_nouns(self) -> set:
        """
        Loads a comprehensive set of inanimate nouns.
        """
        inanimate_nouns = set([
            # Expanded list of inanimate nouns
            "city", "technology", "idea", "concept", "building", "road", "vehicle",
            "computer", "phone", "book", "instrument", "tool", "device", "machine",
            "equipment", "furniture", "material", "substance", "chemical", "metal",
            "plastic", "glass", "fabric", "paper", "water", "air", "fire", "earth",
            "stone", "rock", "mountain", "river", "ocean", "lake", "forest", "desert",
            "planet", "star", "galaxy", "universe", "atom", "molecule", "cell",
            "organ", "tissue", "system", "network", "structure", "algorithm",
            "software", "hardware", "data", "information", "code", "signal",
            "energy", "force", "power", "light", "sound", "color", "shape",
            "size", "temperature", "pressure", "gravity", "magnetism", "electricity",
            "currency", "money", "economy", "market", "industry", "product",
            "service", "policy", "law", "rule", "regulation", "contract",
            "agreement", "document", "report", "plan", "strategy", "project",
            "task", "goal", "objective", "result", "outcome", "effect",
            "impact", "change", "development", "trend", "pattern", "system",
            "method", "approach", "process", "procedure", "practice", "technique",
            "style", "design", "model", "framework", "theory", "hypothesis",
            "experiment", "test", "analysis", "evaluation", "measurement",
            "assessment", "review", "summary", "overview", "description",
            "explanation", "definition", "concept", "idea", "notion", "thought",
            "belief", "value", "principle", "standard", "norm", "criterion",
            "parameter", "variable", "function", "operation", "calculation",
            "equation", "formula", "algorithm", "model", "simulation", "representation"
        ])
        return inanimate_nouns

    def standard_method(self, text: str) -> dict:
        """
        Calculates the inanimacy score based on heuristic linguistic analysis.
        """
        try:
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            inanimate_count = sum(1 for word, tag in pos_tags if tag.startswith('NN') and word in self.inanimate_nouns)
            total_nouns = sum(1 for word, tag in pos_tags if tag.startswith('NN'))

            inanimate_score = (inanimate_count / total_nouns) * 100 if total_nouns else 0
            return {self.name: {'score': inanimate_score, 'error': False}}
        except Exception as e:
            return {self.name: {'score': 0.0, 'error': True, 'error_message': str(e)}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the inanimacy score using NER.
        """
        try:
            ner_results = model(text)
            if not ner_results:
                return {self.name: {'score': 0.0, 'error': False}}

            entity_label_key = 'entity_group' if 'entity_group' in ner_results[0] else 'entity'

            inanimate_labels = {'ORG', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'FAC'}

            inanimate_entities = sum(1 for ent in ner_results if ent[entity_label_key] in inanimate_labels)
            total_entities = len(ner_results)

            inanimate_score = (inanimate_entities / total_entities) * 100 if total_entities else 0
            inanimate_score = max(0.0, min(100.0, inanimate_score))

            return {self.name: {'score': inanimate_score, 'error': False}}
        except Exception as e:
            return {self.name: {'score': 0.0, 'error': True, 'error_message': str(e)}}
'''

