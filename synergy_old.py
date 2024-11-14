# synergy.py

class SynergyCalculator:
    def __init__(self, dimension_scores):
        """
        Initialize the SynergyCalculator with the dimension scores.

        :param dimension_scores: A dictionary where keys are dimension names and values are the scores for each dimension.
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
                'conditions': lambda s: s.get('Affective', 0) > self.threshold and s.get('Cognitive', 0) > self.threshold,
                'comment': (
                    "The high Affective and Cognitive scores suggest that emotional engagement is influencing logical reasoning in this text. "
                    "This synergy is critical in persuasive writing, where emotional appeal is balanced with well-structured argumentation. "
                    "Academic studies have shown that emotional elements can enhance cognitive processing, especially in contexts like advertising, political speeches, and motivational talks. "
                    "(Oxford Academic: https://academic.oup.com) (SpringerLink: https://link.springer.com)."
                )
            },
            {
                'conditions': lambda s: s.get('Cognitive', 0) > self.threshold and s.get('Dynamic', 0) > self.threshold,
                'comment': (
                    "The combination of high Cognitive and Dynamic scores suggests that the text engages with complex, evolving processes or actions. "
                    "This synergy is significant in texts that explain strategic planning, decision-making, or any process requiring thoughtful analysis. "
                    "It's often seen in technical reports, strategic proposals, and scientific studies. "
                    "(SpringerLink: https://link.springer.com)."
                )
            },
            {
                'conditions': lambda s: s.get('Social', 0) > self.threshold and s.get('Cognitive', 0) > self.threshold,
                'comment': (
                    "High Social and Cognitive scores highlight the interaction between social structures and logical thinking. "
                    "This synergy often emerges in sociolinguistics or discourse analysis, where social contexts shape cognitive understanding. "
                    "Texts in education, leadership, or organizational behavior often reflect this interaction. "
                    "(Oxford Academic: https://academic.oup.com)."
                )
            },
            {
                'conditions': lambda s: s.get('Cognitive', 0) > self.threshold and s.get('Time', 0) > self.threshold and s.get('Dynamic', 0) > self.threshold,
                'comment': (
                    "The high Cognitive, Time, and Dynamic scores suggest the text is focused on explaining evolving concepts or events over time in a logically structured way. "
                    "This synergy is essential for texts that describe complex changes or processes, such as in scientific research, historical analysis, or long-term project planning."
                )
            },
            {
                'conditions': lambda s: s.get('Positive', 0) > self.threshold and s.get('Intentionality', 0) > self.threshold and s.get('Novelty', 0) > self.threshold,
                'comment': (
                    "The strong Positive, Intentionality, and Novelty scores indicate that the text is promoting new ideas with clear, positive intentions. "
                    "This combination is often found in visionary or strategic texts where new approaches are framed optimistically and purposefully, such as in innovation strategies or leadership directives."
                )
            },
            {
                'conditions': lambda s: s.get('Negative', 0) > self.threshold and s.get('Cognitive', 0) > self.threshold and s.get('Time', 0) > self.threshold,
                'comment': (
                    "The combination of Negative, Cognitive, and Time-related content suggests the text is focused on negative outcomes or events being evaluated over time. "
                    "This could indicate a critical analysis of failures or challenges, often seen in retrospective evaluations, crisis reporting, or investigative journalism."
                )
            },
            {
                'conditions': lambda s: s.get('Social', 0) > self.threshold and s.get('Place', 0) > self.threshold and s.get('Commonality', 0) > self.threshold,
                'comment': (
                    "High scores in Social, Place, and Commonality suggest the text is focused on shared social experiences tied to specific locations. "
                    "This synergy is relevant in community narratives, geographical identity discussions, or analyses of local culture."
                )
            },
            {
                'conditions': lambda s: s.get('Formality', 0) > self.threshold and s.get('Directness', 0) > self.threshold and s.get('Cognitive', 0) > self.threshold,
                'comment': (
                    "The combination of high Formality, Directness, and Cognitive scores indicates the text is both formally structured and intellectually clear, while also being straightforward. "
                    "This synergy is important for legal documents, professional reports, or formal communication, where clarity, precision, and directness are critical."
                )
            },
            {
                'conditions': lambda s: s.get('Specific', 0) > self.threshold and s.get('Quantitative', 0) > self.threshold and s.get('Time', 0) > self.threshold,
                'comment': (
                    "High Specific, Quantitative, and Time scores suggest the text provides detailed, time-sensitive numerical data. "
                    "This synergy is critical in technical reports, financial analyses, or any context requiring accurate, time-based quantitative information."
                )
            },
            {
                'conditions': lambda s: s.get('Objectivity', 0) > self.threshold and s.get('Novelty', 0) > self.threshold and (s.get('Static', 0) > self.threshold or s.get('Dynamic', 0) > self.threshold),
                'comment': (
                    "The text maintains objectivity while introducing novel concepts, whether static or dynamic. "
                    "This synergy is essential for presenting new ideas or findings while adhering to factual, unbiased reporting, as seen in scientific papers or technical reports."
                )
            },
            {
                'conditions': lambda s: s.get('Affective', 0) > self.threshold and s.get('Social', 0) > self.threshold and s.get('Politeness', 0) > self.threshold,
                'comment': (
                    "The high Affective, Social, and Politeness scores indicate that the text is emotionally engaging while maintaining respectful communication. "
                    "This synergy is useful in analyzing texts involving sensitive social interactions, such as public relations or diplomatic communications."
                )
            },
            # Add more synergy definitions as needed
        ]

        # Iterate through synergy definitions and add comments if conditions are met
        for synergy in synergy_definitions:
            if synergy['conditions'](self.scores):
                self.synergy_comments.append(synergy['comment'])

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

