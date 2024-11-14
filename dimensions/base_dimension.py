# dimensions/base_dimension.py

import nltk

class BaseDimension:
    def __init__(self, name, method='standard'):
        """
        Initialize the BaseDimension with a name and default method.

        :param name: Name of the dimension.
        :param method: Default method ('standard' or 'advanced').
        """
        self.name = name
        self.method = method
        self.setup_nltk()

    def setup_nltk(self):
        """
        Ensures that all necessary NLTK data packages are downloaded.
        """
        required_packages = [
            ('tokenizers/punkt', 'punkt'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
            ('corpora/universal_tagset', 'universal_tagset'),
            ('help/tagsets', 'tagsets'),
            ('corpora/wordnet', 'wordnet'),
            ('corpora/stopwords', 'stopwords'),
            ('corpora/brown', 'brown'),
            ('sentiment/opinion_lexicon', 'opinion_lexicon')
        ]

        for resource_path, package_name in required_packages:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                nltk.download(package_name, quiet=True)

    def calculate_score(self, text, method='standard', model=None):
        """
        Calculates the dimension score based on the selected method.

        :param text: The input text to analyze.
        :param method: 'standard' or 'advanced' method.
        :param model: The model to use for the advanced method.
        :return: A dictionary containing the dimension score.
        """
        if model is None:
            print(f"No model available for '{self.name}'. Using standard method.")
            return self.standard_method(text)
        else:
            try:
                result = self.advanced_method(text, model)
                print(f"Advanced method used for '{self.name}'.")
                return result
            except Exception as e:
                print(f"Error in advanced method for '{self.name}': {e}. Falling back to standard method.")
                return self.standard_method(text)

    def standard_method(self, text):
        """
        Placeholder for the standard method. Must be implemented by subclasses.

        :param text: The input text to analyze.
        :return: A dictionary containing the dimension score.
        """
        raise NotImplementedError("Standard method not implemented.")

    def advanced_method(self, text, model):
        """
        Placeholder for the advanced method. Must be implemented by subclasses.

        :param text: The input text to analyze.
        :param model: The pre-loaded model for advanced analysis.
        :return: A dictionary containing the dimension score.
        """
        raise NotImplementedError("Advanced method not implemented.")

