# dimensions/Place.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class Place(BaseDimension):
    def __init__(self):
        super().__init__(name="Place")
        self.stop_words = set(stopwords.words('english'))
        # Common place-related words
        self.place_words = set(["city", "town", "village", "country", "state", "province",
            "region", "district", "neighborhood", "suburb", "capital",
            "metropolis", "hamlet", "municipality", "county", "shore",
            "beach", "mountain", "river", "lake", "ocean", "sea",
            "forest", "park", "garden", "desert", "valley", "hill",
            "island", "peninsula", "canyon", "cliff", "cove",
            "harbor", "port", "airport", "station", "bridge",
            "road", "street", "avenue", "boulevard", "lane",
            "drive", "trail", "path", "highway", "motorway",
            "freeway", "expressway", "roadway", "route", "intersection",
            "junction", "crossroad", "roundabout", "traffic",
            "light", "sign", "marker", "boundary", "border",
            "territory", "zone", "area", "locale", "location",
            "site", "spot", "point", "position", "place",
            "address", "venue", "facility", "premises", "property",
            "estate", "residence", "dwelling", "home", "house",
            "apartment", "condominium", "office", "business",
            "institution", "establishment", "factory", "plant",
            "warehouse", "store", "shop", "mall", "market",
            "bazaar", "emporium", "cafe", "restaurant", "bar",
            "pub", "club", "hotel", "motel", "hostel",
            "inn", "lodge", "guesthouse", "bedroom", "bathroom",
            "kitchen", "living room", "dining room", "garage",
            "basement", "attic", "garden", "yard", "balcony",
            "terrace", "patio", "deck", "driveway", "parking lot",
            "vehicle", "car", "truck", "bus",
            "train", "subway", "tram", "ferry", "boat",
            "ship", "airplane", "helicopter", "rocket", "spacecraft",
            "satellite", "drone", "bicycle", "motorcycle", "scooter",
            "skateboard", "rollerblades", "carriage", "cart",
            "rickshaw", "buggy", "wagon"])

    def standard_method(self, text: str) -> dict:
        try:
            words = word_tokenize(text.lower())
            total_words = len(words)
            place_count = sum(1 for word in words if word in self.place_words)
            place_score = (place_count / total_words) * 100 if total_words else 0.0
            return {'Place': {'score': place_score, 'error': False}}
        except Exception as e:
            return {'Place': {'score': 0.0, 'error': True, 'error_message': str(e)}}


'''
# dimensions/Place.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from transformers.pipelines import Pipeline
import spacy

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

# Load spaCy model for NER
nlp = spacy.load('en_core_web_sm')

class Place(BaseDimension):
    def __init__(self):
        super().__init__(name="Place")
        # Initialize the set of place-related nouns
        self.place_nouns = self.load_place_nouns()
        # Initialize spatial prepositions
        self.spatial_prepositions = self.load_spatial_prepositions()
        # Compile regex patterns for performance
        self.place_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in self.place_nouns) + r')\b',
            re.IGNORECASE
        )
        self.preposition_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(prep) for prep in self.spatial_prepositions) + r')\b',
            re.IGNORECASE
        )

    def load_place_nouns(self) -> set:
        """
        Loads a predefined set of nouns related to places.

        :return: A set containing place-related nouns.
        """
        # Expanded list of place-related nouns
        place_nouns = {
            "city", "town", "village", "country", "state", "province",
            "region", "district", "neighborhood", "suburb", "capital",
            "metropolis", "hamlet", "municipality", "county", "shore",
            "beach", "mountain", "river", "lake", "ocean", "sea",
            "forest", "park", "garden", "desert", "valley", "hill",
            "island", "peninsula", "canyon", "cliff", "cove",
            "harbor", "port", "airport", "station", "bridge",
            "road", "street", "avenue", "boulevard", "lane",
            "drive", "trail", "path", "highway", "motorway",
            "freeway", "expressway", "roadway", "route", "intersection",
            "junction", "crossroad", "roundabout", "traffic",
            "light", "sign", "marker", "boundary", "border",
            "territory", "zone", "area", "locale", "location",
            "site", "spot", "point", "position", "place",
            "address", "venue", "facility", "premises", "property",
            "estate", "residence", "dwelling", "home", "house",
            "apartment", "condominium", "office", "business",
            "institution", "establishment", "factory", "plant",
            "warehouse", "store", "shop", "mall", "market",
            "bazaar", "emporium", "cafe", "restaurant", "bar",
            "pub", "club", "hotel", "motel", "hostel",
            "inn", "lodge", "guesthouse", "bedroom", "bathroom",
            "kitchen", "living room", "dining room", "garage",
            "basement", "attic", "garden", "yard", "balcony",
            "terrace", "patio", "deck", "driveway", "parking lot",
            "vehicle", "car", "truck", "bus",
            "train", "subway", "tram", "ferry", "boat",
            "ship", "airplane", "helicopter", "rocket", "spacecraft",
            "satellite", "drone", "bicycle", "motorcycle", "scooter",
            "skateboard", "rollerblades", "carriage", "cart",
            "rickshaw", "buggy", "wagon"
        }
        return place_nouns

    def load_spatial_prepositions(self) -> set:
        """
        Loads a predefined set of spatial prepositions.

        :return: A set containing spatial prepositions.
        """
        spatial_prepositions = {
            "in", "on", "at", "by", "near", "beside", "next to",
            "adjacent to", "between", "among", "behind", "beyond",
            "within", "without", "inside", "outside", "onto",
            "toward", "towards", "up", "down", "through", "across",
            "over", "under", "above", "below", "amidst", "amid",
            "throughout", "around", "about", "against", "during",
            "per", "along", "past", "since", "to", "from", "via",
            "with", "including", "excluding", "amongst", "opposite",
            "in front of", "behind", "beyond", "atop", "beneath",
            "underneath", "perpendicular to", "parallel to"
        }
        return spatial_prepositions

    def standard_method(self, text: str) -> dict:
        """
        Calculates the place-relatedness score based on heuristic linguistic analysis.

        :param text: The input text to analyze.
        :return: A dictionary containing the place-relatedness score.
        """
        try:
            words = word_tokenize(text.lower())
            total_words = len(words)
            if total_words == 0:
                return {'Place': {'score': 0.0, 'error': False}}

            place_count = sum(1 for word in words if word in self.place_nouns)
            preposition_count = sum(1 for word in words if word in self.spatial_prepositions)

            # Calculate ratios
            place_ratio = place_count / total_words
            preposition_ratio = preposition_count / total_words

            # Compute place-relatedness score
            # Higher place_ratio and preposition_ratio indicate higher place-relatedness
            # Weighted formula: place_ratio * 0.6 + preposition_ratio * 0.4
            place_score = (place_ratio * 0.6) + (preposition_ratio * 0.4)
            # Scale to 0-100
            place_score *= 100
            # Clamp the score between 0 and 100
            place_score = max(0.0, min(100.0, place_score))

            return {'Place': {'score': place_score, 'error': False}}

        except Exception as e:
            # If an error occurs, set score to 0 and flag error
            return {'Place': {'score': 0.0, 'error': True, 'error_message': str(e)}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the place-relatedness score using a Named Entity Recognition (NER) model.

        :param text: The input text to analyze.
        :param model: The pre-loaded NER model.
        :return: A dictionary containing the place-relatedness score.
        """
        try:
            # Use the NER model to extract entities
            ner_results = model(text)
            if not ner_results:
                return {'Place': {'score': 0.0, 'error': False}}

            total_entities = 0
            place_entities = 0

            # Determine the correct key for entity labels
            entity_label_key = 'entity_group' if 'entity_group' in ner_results[0] else 'entity'

            for entity in ner_results:
                entity_label = entity[entity_label_key]

                # Define criteria for place entities based on entity labels
                # Common labels: 'LOC' (Location), 'GPE' (Geo-Political Entity), 'FAC' (Facility)
                if entity_label in {'LOC', 'GPE', 'FAC'}:
                    place_entities += 1
                total_entities += 1

            if total_entities == 0:
                place_score = 0.0
            else:
                # Calculate the proportion of place entities
                place_proportion = (place_entities / total_entities) * 100
                # Clamp the score between 0 and 100
                place_score = max(0.0, min(100.0, place_proportion))

            return {'Place': {'score': place_score, 'error': False}}

        except Exception as e:
            # If an error occurs, set score to 0 and flag error
            return {'Place': {'score': 0.0, 'error': True, 'error_message': str(e)}}
'''

