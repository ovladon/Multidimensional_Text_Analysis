# dimensions/animate.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize
from transformers.pipelines import Pipeline

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

class Animate(BaseDimension):
    def __init__(self):
        super().__init__(name="Animate")
        self.animal_names = self.load_animal_names()

    def load_animal_names(self):
        """
        Loads a comprehensive list of animal names.
        """
        animal_names = set([
        # Mammals
        'aardvark', 'alpaca', 'anteater', 'antelope', 'ape', 'armadillo', 'baboon',
        'badger', 'bat', 'bear', 'beaver', 'bison', 'boar', 'buffalo', 'bull', 'camel',
        'capybara', 'caribou', 'cat', 'cattle', 'cheetah', 'chimpanzee', 'chipmunk',
        'cougar', 'cow', 'coyote', 'deer', 'dog', 'dolphin', 'donkey', 'elephant',
        'elk', 'fox', 'giraffe', 'goat', 'gorilla', 'hamster', 'hare', 'hedgehog',
        'hippopotamus', 'horse', 'hyena', 'impala', 'jaguar', 'kangaroo', 'koala',
        'lemur', 'leopard', 'lion', 'llama', 'lynx', 'manatee', 'mink', 'mole', 'mongoose',
        'monkey', 'moose', 'mouse', 'mule', 'otter', 'ox', 'panda', 'panther', 'platypus',
        'polar bear', 'porcupine', 'rabbit', 'raccoon', 'rat', 'reindeer', 'rhinoceros',
        'seal', 'sheep', 'skunk', 'sloth', 'squirrel', 'tiger', 'walrus', 'weasel', 'whale',
        'wolf', 'wombat', 'yak', 'zebra',

        # Birds
        'albatross', 'crow', 'dove', 'duck', 'eagle', 'falcon', 'flamingo', 'goose',
        'hawk', 'heron', 'hummingbird', 'ibis', 'kingfisher', 'magpie', 'nightingale',
        'ostrich', 'owl', 'parrot', 'peacock', 'pelican', 'penguin', 'pigeon', 'quail',
        'raven', 'robin', 'sparrow', 'stork', 'swan', 'vulture', 'woodpecker',

        # Reptiles and amphibians
        'alligator', 'anaconda', 'boa', 'chameleon', 'cobra', 'crocodile', 'frog',
        'gecko', 'iguana', 'lizard', 'python', 'salamander', 'snake', 'toad', 'tortoise', 'turtle',

        # Fish and invertebrates
        'anchovy', 'anglerfish', 'barracuda', 'bass', 'bluefish', 'carp', 'catfish',
        'clownfish', 'cod', 'eel', 'flounder', 'goldfish', 'haddock', 'lobster',
        'mackerel', 'octopus', 'pike', 'ray', 'salmon', 'sardine', 'shark',
        'shrimp', 'snapper', 'trout', 'tuna', 'whiting',

        # Insects
        'bee', 'beetle', 'butterfly', 'centipede', 'cricket', 'dragonfly', 'firefly',
        'fly', 'grasshopper', 'hornet', 'ladybug', 'moth', 'scorpion', 'spider',
        'termite', 'wasp', 'worm',

        # People (included separately in the standard method)
        'person', 'people', 'man', 'woman', 'child', 'boy', 'girl', 'family', 'friend',
        'individual', 'adult', 'parent', 'mother', 'father', 'brother', 'sister',
        'husband', 'wife', 'uncle', 'aunt', 'cousin', 'nephew', 'niece', 'grandparent',
        'grandmother', 'grandfather', 'employee', 'employer', 'worker', 'colleague',
        'customer', 'client', 'patient', 'doctor', 'nurse', 'teacher', 'student',
        'police', 'officer', 'soldier', 'firefighter'
        ])
        return animal_names

    def standard_method(self, text):
        """
        Standard method uses a predefined list of animate nouns to count occurrences.
        """
        tokens = word_tokenize(text.lower())
        animate_count = sum(1 for word in tokens if word in self.animal_names)
        total_words = len(tokens)
        score = (animate_count / total_words) * 100 if total_words else 0
        return {self.name: {'score': score, 'error': False}}

    def advanced_method(self, text, model: Pipeline):
        """
        Advanced method uses NER to identify animate entities.
        """
        try:
            # Run NER on the text to detect entities
            entities = model(text)
            if not entities:
                return {self.name: {'score': 0.0, 'error': False}}

            # Determine the correct key for entity labels
            entity_label_key = 'entity_group' if 'entity_group' in entities[0] else 'entity'

            # Define animate entity labels
            animate_entity_labels = {'PER', 'PERSON', 'ANIMAL'}

            # Count animate entities detected by NER
            animate_entities = sum(1 for ent in entities if ent[entity_label_key] in animate_entity_labels)

            # Count animals from the predefined list
            tokens = word_tokenize(text.lower())
            animal_count = sum(1 for word in tokens if word in self.animal_names)

            # Calculate the total score
            total_count = animate_entities + animal_count
            total_words = len(tokens)
            score = (total_count / total_words) * 100 if total_words else 0
            return {self.name: {'score': score, 'error': False}}
        except Exception as e:
            return {self.name: {'score': 0.0, 'error': True, 'error_message': str(e)}}



