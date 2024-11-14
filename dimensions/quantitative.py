# dimensions/Quantitative.py

from .base_dimension import BaseDimension
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from transformers.pipelines import Pipeline
import math

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

class Quantitative(BaseDimension):
    def __init__(self, method='standard'):
        super().__init__(name="Quantitative", method=method)
        # Initialize comprehensive sets of quantitative indicators
        self.quantitative_adjectives = self.load_quantitative_adjectives()
        self.statistical_terms = self.load_statistical_terms()
        self.measurement_units = self.load_measurement_units()
        # Compile regex patterns for performance
        self.number_pattern = re.compile(r'\b\d+(\.\d+)?\b')  # Matches integers and decimals
        self.quantitative_adj_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in self.quantitative_adjectives) + r')\b',
            re.IGNORECASE
        )
        self.statistical_term_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(term) for term in self.statistical_terms) + r')\b',
            re.IGNORECASE
        )
        self.measurement_unit_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(unit) for unit in self.measurement_units) + r')\b',
            re.IGNORECASE
        )

    def load_quantitative_adjectives(self) -> set:
        """
        Loads a comprehensive set of adjectives that indicate quantitative descriptions.

        :return: A set containing quantitative adjectives.
        """
        quantitative_adjectives = {
            "numerous", "several", "various", "multiple", "many", "few",
            "significant", "substantial", "considerable", "extensive",
            "increased", "decreased", "elevated", "reduced", "enhanced",
            "diminished", "boosted", "lowered", "raised", "high", "low",
            "greater", "lesser", "additional", "extra", "further", "more",
            "less", "some", "all", "total", "complete", "partial", "major",
            "minor", "average", "median", "mode", "maximum", "minimum",
            "higher", "lower", "best", "worst", "primary", "secondary",
            "tertiary", "quantitative", "qualitative", "statistical",
            "analytic", "analytical", "empirical", "theoretical", "predictive",
            "descriptive", "exploratory", "confirmatory", "correlational",
            "causal", "comparative", "longitudinal", "cross-sectional"
        }
        return quantitative_adjectives

    def load_statistical_terms(self) -> set:
        """
        Loads a comprehensive set of statistical terms.

        :return: A set containing statistical terms.
        """
        statistical_terms = {
            "regression", "ANOVA", "t-test", "chi-square", "factor analysis",
            "principal component analysis", "cluster analysis",
            "time series", "hypothesis", "null hypothesis", "alternative hypothesis",
            "p-value", "confidence interval", "standard error", "degrees of freedom",
            "beta coefficient", "alpha level", "power analysis", "effect size",
            "multivariate", "univariate", "bivariate", "covariance", "standard deviation",
            "variance", "skewness", "kurtosis", "z-score", "t-score",
            "logistic regression", "linear regression", "hierarchical regression",
            "stepwise regression", "moderator", "mediator", "interaction",
            "correlation coefficient", "Pearson's r", "Spearman's rho",
            "Kendall's tau", "chi-squared test", "F-test", "likelihood ratio",
            "Bayesian statistics", "frequentist statistics", "Monte Carlo simulation",
            "bootstrap", "permutation test", "parametric test", "non-parametric test"
        }
        return statistical_terms

    def load_measurement_units(self) -> set:
        """
        Loads a comprehensive set of measurement units.

        :return: A set containing measurement units.
        """
        measurement_units = {
            "meter", "kilometer", "mile", "foot", "inch", "centimeter",
            "millimeter", "gram", "kilogram", "pound", "ounce", "liter",
            "milliliter", "gallon", "quart", "pint", "cup", "tablespoon",
            "teaspoon", "degree", "celsius", "fahrenheit", "kelvin",
            "percent", "%", "dollar", "euro", "yen", "pound sterling",
            "bitcoin", "megabyte", "gigabyte", "terabyte", "hertz",
            "kilohertz", "megahertz", "gigahertz", "newton", "joule",
            "watt", "pascal", "volt", "ampere", "ohm", "farad",
            "siemens", "weber", "tesla", "lux", "lumen", "candela",
            "mole", "becquerel", "gray", "sievert", "katal", "bit",
            "byte", "gigahertz", "megahertz", "nanometer", "micrometer",
            "nanosecond", "microsecond", "millisecond", "second", "minute",
            "hour", "day", "week", "month", "year", "decade", "century",
            "millennium"
        }
        return measurement_units

    def standard_method(self, text: str) -> dict:
        """
        Calculates the quantitative score based on heuristic linguistic analysis.

        :param text: The input text to analyze.
        :return: A dictionary containing the quantitative score.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Quantitative': {'score': 0.0, 'error': False}}

            total_numbers = 0
            total_quantitative_adjs = 0
            total_statistical_terms = 0
            total_measurement_units = 0
            total_words = 0
            total_sentences = len(sentences)

            for sentence in sentences:
                words = word_tokenize(sentence)
                if not words:
                    continue

                total_words += len(words)

                # Count numerical expressions
                number_matches = self.number_pattern.findall(sentence)
                total_numbers += len(number_matches)

                # Count quantitative adjectives
                quantitative_adj_matches = self.quantitative_adj_pattern.findall(sentence)
                total_quantitative_adjs += len(quantitative_adj_matches)

                # Count statistical terms
                statistical_term_matches = self.statistical_term_pattern.findall(sentence)
                total_statistical_terms += len(statistical_term_matches)

                # Count measurement units
                measurement_unit_matches = self.measurement_unit_pattern.findall(sentence)
                total_measurement_units += len(measurement_unit_matches)

            if total_words == 0:
                quantitative_score = 0.0
            else:
                # Calculate ratios
                number_ratio = total_numbers / total_words
                quantitative_adj_ratio = total_quantitative_adjs / total_words
                statistical_term_ratio = total_statistical_terms / total_words
                measurement_unit_ratio = total_measurement_units / total_words

                # Compute quantitative score
                # Higher ratios indicate higher quantitative content
                # Weighted formula: number_ratio * 0.4 + quantitative_adj_ratio * 0.3 +
                # statistical_term_ratio * 0.2 + measurement_unit_ratio * 0.1
                quantitative_score = (number_ratio * 0.4) + \
                                     (quantitative_adj_ratio * 0.3) + \
                                     (statistical_term_ratio * 0.2) + \
                                     (measurement_unit_ratio * 0.1)
                # Scale to 0-100
                quantitative_score *= 100
                # Clamp the score between 0 and 100
                quantitative_score = max(0.0, min(100.0, quantitative_score))

            return {'Quantitative': {'score': quantitative_score, 'error': False}}

        except Exception as e:
            # If an error occurs, set score to 0 and flag error
            return {'Quantitative': {'score': 0.0, 'error': True}}

    def advanced_method(self, text: str, model: Pipeline) -> dict:
        """
        Calculates the quantitative score using a language model (LLM) suitable for quantitative analysis.
        Combines model predictions with heuristic-based assessments to enhance accuracy.

        :param text: The input text to analyze.
        :param model: The pre-loaded language model for quantitative analysis.
        :return: A dictionary containing the quantitative score.
        """
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return {'Quantitative': {'score': 0.0, 'error': False}}

            quantitative_predictions = 0.0
            total_predictions = 0.0

            for sentence in sentences:
                # Ensure sentence is within the model's maximum token limit
                if len(word_tokenize(sentence)) > 512:
                    continue  # Skip overly long sentences

                predictions = model(sentence)

                if not predictions:
                    continue

                # Assuming the model returns a list of dicts with 'label' and 'score'
                # Example labels for zero-shot classification: 'Quantitative', 'Non-Quantitative'
                top_prediction = predictions[0]
                label = top_prediction['label'].lower()
                score = top_prediction['score']

                if label == 'quantitative':
                    quantitative_predictions += score
                elif label == 'non-quantitative':
                    quantitative_predictions += (1 - score)  # Treat 'Non-Quantitative' as inverse of 'Quantitative'
                total_predictions += score if label == 'quantitative' else (1 - score)

            if total_predictions == 0.0:
                # If no model predictions, fallback to standard method
                standard_results = self.standard_method(text)
                return standard_results

            # Calculate average quantitative score from model
            average_quantitative = (quantitative_predictions / total_predictions) * 100

            # To ensure advanced_method outperforms standard_method, combine with heuristic counts
            standard_results = self.standard_method(text)
            standard_score = standard_results.get('Quantitative', {}).get('score', 0.0)

            # Weighted combination: model's average + standard_method's score
            # Adjust weights as needed; here, giving model 70% and standard 30%
            combined_quantitative = (average_quantitative * 0.7) + (standard_score * 0.3)
            combined_quantitative = max(0.0, min(100.0, combined_quantitative))

            return {'Quantitative': {'score': combined_quantitative, 'error': False}}

        except Exception as e:
            # If a suitable model is not available, fallback to combining standard method with heuristic-based assessments
            try:
                standard_results = self.standard_method(text)
                return standard_results
            except Exception as inner_e:
                return {'Quantitative': {'score': 0.0, 'error': True}}

