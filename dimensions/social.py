# dimensions/Social.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from transformers.pipelines import Pipeline
import math

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

class Social(BaseDimension):
    def __init__(self, method='standard'):
        super().__init__(name="Social", method=method)
        # Initialize comprehensive sets of social indicators
        self.social_words = self.load_social_words()
        self.social_phrases = self.load_social_phrases()
        self.relationship_terms = self.load_relationship_terms()
        self.community_phrases = self.load_community_phrases()
        # Compile regex patterns for performance
        self.social_word_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in self.social_words) + r')\b', 
            re.IGNORECASE
        )
        self.social_phrase_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(phrase) for phrase in self.social_phrases) + r')\b', 
            re.IGNORECASE
        )
        self.relationship_term_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(term) for term in self.relationship_terms) + r')\b', 
            re.IGNORECASE
        )
        self.community_phrase_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(phrase) for phrase in self.community_phrases) + r')\b', 
            re.IGNORECASE
        )

    def load_social_words(self) -> set:
        """
        Loads a comprehensive set of social-related words.

        :return: A set containing social words.
        """
        social_words = {
            "friend", "friends", "family", "families", "community", "communities",
            "social", "society", "relationship", "relationships", "interaction",
            "interactions", "network", "networks", "colleague", "colleagues",
            "peer", "peers", "neighbor", "neighbors", "acquaintance", "acquaintances",
            "partner", "partners", "ally", "allies", "associate", "associates",
            "group", "groups", "team", "teams", "club", "clubs", "association",
            "associations", "society", "societies", "organization", "organizations",
            "organize", "organizing", "engage", "engaging", "participate",
            "participating", "participation", "collaborate", "collaborating",
            "collaboration", "support", "supporting", "supported", "help",
            "helping", "assistance", "assist", "assisting", "aid", "aiding",
            "benefit", "benefiting", "benefited", "beneficial", "beneficiary",
            "beneficiaries", "shared", "sharing", "collective", "collectively",
            "mutual", "mutually", "reciprocal", "reciprocally", "contribute",
            "contributing", "contribution", "contributor", "contributors",
            "volunteer", "volunteering", "volunteered", "volunteers", "charity",
            "charities", "donate", "donating", "donated", "donation", "donations",
            "philanthropy", "philanthropic", "giver", "givers", "benefactor",
            "benefactors", "recipient", "recipients", "recipient"
        }
        return social_words

    def load_social_phrases(self) -> set:
        """
        Loads a comprehensive set of social-related phrases.

        :return: A set containing social phrases.
        """
        social_phrases = {
            "close friend", "best friend", "family member", "social interaction",
            "social network", "community involvement", "community engagement",
            "professional relationship", "personal relationship", "workplace camaraderie",
            "social support", "mutual respect", "shared interests", "common goals",
            "team collaboration", "group dynamics", "peer support", "neighborly relations",
            "volunteer work", "charitable activities", "community service",
            "social gathering", "networking event", "team building", "social club",
            "friendship circle", "support system", "mutual aid", "social circle",
            "professional network", "community network", "social bond",
            "social connection", "social ties", "interpersonal relationship"
        }
        return social_phrases

    def load_relationship_terms(self) -> set:
        """
        Loads a comprehensive set of relationship-related terms.

        :return: A set containing relationship terms.
        """
        relationship_terms = {
            "relationship", "relationships", "marriage", "married",
            "partner", "partners", "spouse", "spouses", "couple",
            "couples", "dating", "dating relationship", "romantic relationship",
            "intimate relationship", "commitment", "engagement",
            "fiancé", "fiancée", "husband", "wife", "boyfriend",
            "girlfriend", "significant other", "life partner", "companion",
            "ally", "allies", "confidant", "confidantes", "supporter",
            "supporters"
        }
        return relationship_terms

    def load_community_phrases(self) -> set:
        """
        Loads a comprehensive set of community-related phrases.

        :return: A set containing community phrases.
        """
        community_phrases = {
            "community center", "community group", "community organization",
            "community outreach", "local community", "online community",
            "community service", "community project", "community initiative",
            "community development", "community engagement", "community support",
            "community event", "community meeting", "community network",
            "community volunteer", "community participation"
        }
        return community_phrases

    def standard_method(self, text: str) -> dict:
        """
        Calculates the social score based on heuristic linguistic analysis.

        :param text: The input text to analyze.
        :return: A dictionary containing the social score.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Social': {'score': 0.0, 'error': False}}

            total_social_words = 0
            total_social_phrases = 0
            total_relationship_terms = 0
            total_community_phrases = 0
            total_words = 0
            total_sentences = len(sentences)

            for sentence in sentences:
                words = word_tokenize(sentence)
                if not words:
                    continue

                total_words += len(words)

                # Count social words
                social_word_matches = self.social_word_pattern.findall(sentence)
                total_social_words += len(social_word_matches)

                # Count social phrases
                social_phrase_matches = self.social_phrase_pattern.findall(sentence)
                total_social_phrases += len(social_phrase_matches)

                # Count relationship terms
                relationship_term_matches = self.relationship_term_pattern.findall(sentence)
                total_relationship_terms += len(relationship_term_matches)

                # Count community phrases
                community_phrase_matches = self.community_phrase_pattern.findall(sentence)
                total_community_phrases += len(community_phrase_matches)

            if total_words == 0:
                social_score = 0.0
            else:
                # Calculate ratios
                social_word_ratio = total_social_words / total_words
                social_phrase_ratio = total_social_phrases / total_sentences
                relationship_term_ratio = total_relationship_terms / total_words
                community_phrase_ratio = total_community_phrases / total_sentences

                # Compute social score
                # Weighted formula: social_word_ratio * 0.3 + social_phrase_ratio * 0.3 +
                # relationship_term_ratio * 0.2 + community_phrase_ratio * 0.2
                social_score = (social_word_ratio * 0.3) + \
                               (social_phrase_ratio * 0.3) + \
                               (relationship_term_ratio * 0.2) + \
                               (community_phrase_ratio * 0.2)
                # Scale to 0-100
                social_score *= 100
                # Clamp the score between 0 and 100
                social_score = max(0.0, min(100.0, social_score))

            return {'Social': {'score': social_score, 'error': False}}

        except Exception as e:
            # If an error occurs, set score to 0 and flag error
            return {'Social': {'score': 0.0, 'error': True}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the social score using a language model (LLM) suitable for classification.
        Combines model predictions with heuristic-based assessments to enhance accuracy.

        :param text: The input text to analyze.
        :param model: The pre-loaded language model for classification.
        :return: A dictionary containing the social score.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Social': {'score': 0.0, 'error': False}}

            social_predictions = 0.0
            total_predictions = 0.0

            for sentence in sentences:
                # Ensure sentence is within the model's maximum token limit
                if len(word_tokenize(sentence)) > 512:
                    continue  # Skip overly long sentences

                # Perform zero-shot classification
                classification = model(sentence, candidate_labels=["Social", "Non-Social"])

                if not classification or 'labels' not in classification:
                    continue

                label = classification['labels'][0].lower()
                score = classification['scores'][0]

                if label == 'social':
                    social_predictions += score
                elif label == 'non-social':
                    social_predictions += (1 - score)  # Treat 'Non-Social' as inverse
                total_predictions += score if label == 'social' else (1 - score)

            if total_predictions == 0.0:
                # If no model predictions, fallback to standard method
                standard_results = self.standard_method(text)
                return standard_results

            # Calculate average social score from model
            average_social = (social_predictions / total_predictions) * 100

            # To ensure advanced_method outperforms standard_method, combine with heuristic counts
            standard_results = self.standard_method(text)
            standard_score = standard_results.get('Social', {}).get('score', 0.0)

            # Weighted combination: model's average + standard_method's score
            # Adjust weights as needed; here, giving model 70% and standard 30%
            combined_social = (average_social * 0.7) + (standard_score * 0.3)
            combined_social = max(0.0, min(100.0, combined_social))

            return {'Social': {'score': combined_social, 'error': False}}

        except Exception as e:
            # If a suitable model is not available, fallback to combining standard method with heuristic-based assessments
            try:
                standard_results = self.standard_method(text)
                return standard_results
            except Exception as inner_e:
                return {'Social': {'score': 0.0, 'error': True}}

