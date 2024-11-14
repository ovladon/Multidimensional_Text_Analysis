# dimensions/commonality.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
from collections import Counter
from transformers import pipeline, Pipeline
import math
import os
import pickle

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('brown', quiet=True)

class Commonality(BaseDimension):
    def __init__(self):
        super().__init__(name="Commonality")
        self.word_freq = self.load_word_frequencies()
        self.entity_freq = self.load_entity_frequencies()

    def load_word_frequencies(self):
        """
        Loads word frequencies from the Brown corpus.
        """
        words = brown.words()
        freq_dist = Counter(word.lower() for word in words)
        return freq_dist

    def load_entity_frequencies(self):
        """
        Calculates entity frequencies from the Brown corpus using a NER model.
        Results are cached to avoid reprocessing.
        """
        cache_file = 'entity_frequencies.pkl'

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                entity_freq_dist = pickle.load(f)
            print("Loaded entity frequencies from cache.")
            return entity_freq_dist
        else:
            try:
                # Initialize NER pipeline
                ner_model = pipeline(
                    'ner',
                    model='dbmdz/bert-large-cased-finetuned-conll03-english',
                    aggregation_strategy='simple'
                )
                sentences = brown.sents()
                entities = []

                # Process sentences in batches to improve efficiency
                batch_size = 100
                num_sentences = len(sentences)
                for i in range(0, num_sentences, batch_size):
                    batch_sentences = [' '.join(sent) for sent in sentences[i:i+batch_size]]
                    batch_text = ' '.join(batch_sentences)
                    ner_results = ner_model(batch_text)

                    for entity in ner_results:
                        entity_text = entity['word'].lower()
                        # Clean entity text (remove non-alphabetic characters)
                        entity_text = ''.join(filter(str.isalpha, entity_text))
                        if entity_text:  # Ensure it's not empty after cleaning
                            entities.append(entity_text)

                entity_freq_dist = Counter(entities)

                # Cache the results
                with open(cache_file, 'wb') as f:
                    pickle.dump(entity_freq_dist, f)

                print("Calculated and cached entity frequencies.")
                return entity_freq_dist
            except Exception as e:
                print(f"Error loading entity frequencies: {e}")
                return Counter()

    def standard_method(self, text: str) -> dict:
        """
        Calculates the commonality score based on word frequency.
        """
        try:
            tokens = word_tokenize(text.lower())

            if not tokens:
                return {'Commonality': {'score': 0.0, 'error': False}}

            sum_log_freq = 0
            count = 0
            for word in tokens:
                freq = self.word_freq.get(word, 1)
                if freq > 0:
                    sum_log_freq += math.log(freq)
                    count += 1

            if count == 0:
                return {'Commonality': {'score': 0.0, 'error': False}}

            avg_log_freq = sum_log_freq / count
            max_log_freq = math.log(max(self.word_freq.values()))
            min_nonzero_freq = min(freq for freq in self.word_freq.values() if freq > 0)
            min_log_freq = math.log(min_nonzero_freq)

            normalized_score = ((avg_log_freq - min_log_freq) / (max_log_freq - min_log_freq)) * 100
            normalized_score = max(0.0, min(100.0, normalized_score))

            return {'Commonality': {'score': normalized_score, 'error': False}}
        except Exception as e:
            return {'Commonality': {'score': 0.0, 'error': True, 'error_message': str(e)}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the commonality score based on entity frequency using a NER model.
        """
        try:
            ner_results = model(text)
            if not ner_results:
                return {'Commonality': {'score': 0.0, 'error': False}}

            entities_in_text = []
            for entity in ner_results:
                entity_text = entity['word'].lower()
                # Clean entity text
                entity_text = ''.join(filter(str.isalpha, entity_text))
                if entity_text:
                    entities_in_text.append(entity_text)

            if not entities_in_text:
                return {'Commonality': {'score': 0.0, 'error': False}}

            sum_log_freq = 0
            count = 0
            for entity in entities_in_text:
                freq = self.entity_freq.get(entity, 1)
                if freq > 0:
                    sum_log_freq += math.log(freq)
                    count += 1

            if count == 0:
                return {'Commonality': {'score': 0.0, 'error': False}}

            max_freq = max(self.entity_freq.values())
            min_nonzero_freq = min(freq for freq in self.entity_freq.values() if freq > 0)

            avg_log_freq = sum_log_freq / count
            max_log_freq = math.log(max_freq)
            min_log_freq = math.log(min_nonzero_freq)

            normalized_score = ((avg_log_freq - min_log_freq) / (max_log_freq - min_log_freq)) * 100
            normalized_score = max(0.0, min(100.0, normalized_score))

            return {'Commonality': {'score': normalized_score, 'error': False}}

        except Exception as e:
            return {'Commonality': {'score': 0.0, 'error': True, 'error_message': str(e)}}

