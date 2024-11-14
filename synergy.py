# synergy.py

class SynergyCalculator:
    def __init__(self, dimension_scores):
        """
        Initialize the SynergyCalculator with the dimension scores.

        :param dimension_scores: A dictionary where keys are dimension names and values are dictionaries containing 'score' and 'error' keys.
        """
        self.scores = dimension_scores
        self.threshold = 60  # Threshold for synergy relevance
        self.synergy_comments = []
        self.expected_dimensions = [
            'Affective', 'Cognitive', 'Dynamic', 'Social', 'Novelty',
            'Time', 'Positive', 'Negative', 'Intentionality', 'Formality',
            'Specific', 'Static', 'Quantitative', 'Politeness',
            # Add other dimensions as necessary
        ]

    def detect_synergies(self):
        """
        Detects relevant synergies based on the dimension scores.
        Adds comments explaining each relevant synergy.
        """
        # Check for missing dimensions
        missing_dimensions = [dim for dim in self.expected_dimensions if dim not in self.scores]
        if missing_dimensions:
            self.synergy_comments.append(
                f"**Warning:** Missing dimension scores for {', '.join(missing_dimensions)}. "
                "Synergy analysis may be incomplete."
            )

        # Define synergy conditions and corresponding comments
        synergy_definitions = [
            {
                'conditions': lambda s: s.get('Affective', {}).get('score', 0) > self.threshold and s.get('Cognitive', {}).get('score', 0) > self.threshold,
                'comment': (
                    "The high Affective and Cognitive scores suggest that emotional engagement is influencing logical reasoning in this text. "
                    "This synergy is critical in persuasive writing, where emotional appeal is balanced with well-structured argumentation. "
                    "Academic studies have shown that emotional elements can enhance cognitive processing, especially in contexts like advertising, political speeches, and motivational talks. "
                    "(Oxford Academic: https://academic.oup.com) (SpringerLink: https://link.springer.com)."
                )
            },
            {
                'conditions': lambda s: s.get('Cognitive', {}).get('score', 0) > self.threshold and s.get('Dynamic', {}).get('score', 0) > self.threshold,
                'comment': (
                    "The combination of high Cognitive and Dynamic scores suggests that the text engages with complex, evolving processes or actions. "
                    "This synergy is significant in texts that explain strategic planning, decision-making, or any process requiring thoughtful analysis. "
                    "It's often seen in technical reports, strategic proposals, and scientific studies. "
                    "(SpringerLink: https://link.springer.com)."
                )
            },
            # ... include all synergy definitions with correct access to 'score'
            # Example for another synergy
            {
                'conditions': lambda s: s.get('Social', {}).get('score', 0) > self.threshold and s.get('Cognitive', {}).get('score', 0) > self.threshold,
                'comment': (
                    "High Social and Cognitive scores highlight the interaction between social structures and logical thinking. "
                    "This synergy often emerges in sociolinguistics or discourse analysis, where social contexts shape cognitive understanding. "
                    "Texts in education, leadership, or organizational behavior often reflect this interaction. "
                    "(Oxford Academic: https://academic.oup.com)."
                )
            },
            # Add all other synergy definitions similarly
        ]

        # Iterate through synergy definitions and add comments if conditions are met
        for synergy in synergy_definitions:
            try:
                if synergy['conditions'](self.scores):
                    self.synergy_comments.append(synergy['comment'])
            except Exception as e:
                self.synergy_comments.append(f"**Error in synergy detection:** {e}")

    def get_relevant_synergies(self):
        """
        Detects and returns the relevant synergy comments.

        :return: A list of strings, each representing a relevant synergy and its explanation.
        """
        self.detect_synergies()
        return self.synergy_comments


def generate_synergy_report(dimension_scores):
    """
    Integrates synergy analysis into the final report.

    :param dimension_scores: A dictionary of dimension scores.
    :return: A final report string that includes relevant synergies.
    """
    synergy_calculator = SynergyCalculator(dimension_scores)
    relevant_synergies = synergy_calculator.get_relevant_synergies()

    if relevant_synergies:
        report = "## Synergy Analysis:\n" + "\n\n".join(relevant_synergies)
    else:
        report = "## Synergy Analysis:\nNo significant synergies detected."

    return report

