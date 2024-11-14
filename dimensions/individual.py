# dimensions/Individual.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

class Individual(BaseDimension):
    def __init__(self):
        super().__init__(name="Individual")
        self.individual_words = set(["i", "me", "my", "mine", "myself",
            "you", "your", "yours", "yourself", "yourselves",
            "he", "him", "his", "himself",
            "she", "her", "hers", "herself",
            "we", "us", "our", "ours", "ourselves",
            "they", "them", "their", "theirs", "themselves"])

    def standard_method(self, text: str) -> dict:
        try:
            words = word_tokenize(text)
            total_words = len(words)
            individual_count = sum(1 for word in words if word.lower() in self.individual_words)
            individual_score = (individual_count / total_words) * 100 if total_words else 0.0
            return {'Individual': {'score': individual_score, 'error': False}}
        except Exception as e:
            return {'Individual': {'score': 0.0, 'error': True, 'error_message': str(e)}}


'''
# dimensions/individual.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize
import re
from transformers.pipelines import Pipeline

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

class Individual(BaseDimension):
    def __init__(self):
        super().__init__(name="Individual")
        self.personal_pronouns = self.load_personal_pronouns()
        self.common_first_names = self.load_common_first_names()
        self.pronoun_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, self.personal_pronouns)) + r')\b', re.IGNORECASE)
        self.name_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, self.common_first_names)) + r')\b', re.IGNORECASE)

    def load_personal_pronouns(self) -> set:
        """
        Loads personal pronouns.
        """
        personal_pronouns = {
            "i", "me", "my", "mine", "myself",
            "you", "your", "yours", "yourself", "yourselves",
            "he", "him", "his", "himself",
            "she", "her", "hers", "herself",
            "we", "us", "our", "ours", "ourselves",
            "they", "them", "their", "theirs", "themselves"
        }
        return personal_pronouns

    def load_common_first_names(self) -> set:
        """
        Loads a predefined set of common first names.
        :return: A set containing common first names.
        """
        common_first_names = {
            "john", "jane", "michael", "mary", "robert", "patricia",
            "james", "linda", "william", "barbara", "david", "elizabeth",
            "richard", "jennifer", "charles", "maria", "joseph", "susan",
            "thomas", "margaret", "christopher", "dorothy", "daniel", "lisa",
            "paul", "nancy", "mark", "karen", "donald", "betty", "kenneth",
            "helen", "steven", "sandra", "edward", "diane", "brian", "ruth",
            "ronald", "shirley", "anthony", "cynthia", "kevin", "angela",
            "jason", "melissa", "george", "deborah", "timothy", "stephanie",
            "larry", "jessica", "jeffrey", "katherine", "gary", "martha",
            "nicholas", "danielle", "eric", "donna", "jonathan", "catherine",
            "justin", "kayla", "scott", "kimberly", "brandon", "amanda",
            "gregory", "emily", "jose", "melanie", "patrick", "nicole",
            "aaron", "victoria", "henry", "rebecca", "frank", "laura",
            "raymond", "julia", "jack", "samantha", "dennis", "danielle",
            "walter", "karen", "tyler", "tiffany", "harold", "jacqueline",
            "lawrence", "lillian", "philip", "christine", "christian", "emma",
            "sean", "grace", "joshua", "morgan"
        }
        return common_first_names

    def calculate_score(self, text, method='standard', model=None):
        if method == 'standard':
            return self.standard_method(text)
        elif method == 'advanced':
            if model is None:
                raise ValueError("Advanced method requires a pre-loaded model.")
            return self.advanced_method(text, model)
        else:
            raise ValueError(f"Unknown method: {method}")

    def standard_method(self, text: str) -> dict:
        """
        Calculates the individuality score based on pronouns and names.
        """
        try:
            words = word_tokenize(text.lower())
            total_words = len(words)
            if total_words == 0:
                return {self.name: {'score': 0.0, 'error': False}}

            pronoun_count = len(self.pronoun_pattern.findall(text))
            name_count = len(self.name_pattern.findall(text))

            individual_score = ((pronoun_count + name_count) / total_words) * 100
            return {self.name: {'score': individual_score, 'error': False}}
        except Exception as e:
            return {self.name: {'score': 0.0, 'error': True, 'error_message': str(e)}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the individuality score using NER.
        """
        try:
            entities = model(text)
            if not entities:
                return {self.name: {'score': 0.0, 'error': False}}

            entity_label_key = 'entity_group' if 'entity_group' in entities[0] else 'entity'

            person_entities = sum(1 for ent in entities if ent[entity_label_key] in ['PER', 'PERSON'])
            total_entities = len(entities)

            individual_score = (person_entities / total_entities) * 100 if total_entities else 0
            return {self.name: {'score': individual_score, 'error': False}}
        except Exception as e:
            return {self.name: {'score': 0.0, 'error': True, 'error_message': str(e)}}
'''

