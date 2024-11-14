# Multidimensional_Text_Analysis
An overall view of textual content.
Run python3 start.py in a terminal and the app will run on a local

Multidimensional Text Analysis

An overall view of textual content using a customizable multidimensional approach.
Overview

Multidimensional Text Analysis is a tool designed to provide a comprehensive analysis of text across multiple dimensions, offering users a "bird's eye view" of the general nature and characteristics of their text data. By evaluating texts across various empirically grounded dimensions and identifying synergies between them, the tool reveals deeper insights that are not apparent when dimensions are considered in isolation.

The system is customizable, allowing users to select or define the dimensions most relevant to their needs. It operates locally without requiring an internet connection, ensuring data privacy and security.
Features

    Multidimensional Analysis: Evaluate text across multiple dimensions such as Affective, Cognitive, Time, Place, Dynamic, Static, Animate, Inanimate, and more.
    Synergy Detection: Identify and interpret synergies between dimensions to uncover deeper insights.
    Customizable Dimensions: Add or adjust dimensions to tailor the analysis to specific requirements.
    Resource Adaptability: Operates in both high-performance and resource-constrained environments by choosing between standard and advanced analysis methods.
    Visualization: Generates radar charts and comprehensive reports for easy interpretation of results.
    Local Execution: Runs entirely offline to maintain confidentiality and control over your data.

Proposed Dimensions

Some of the key dimensions analyzed by the tool include:

    Affective: Measures the emotional content expressed in the text.
    Cognitive: Assesses the presence of cognitive processes such as thinking and reasoning.
    Time: Identifies temporal references.
    Place: Detects spatial references.
    Dynamic vs. Static: Evaluates actions, processes, or changes versus states or conditions that are unchanging.
    Animate vs. Inanimate: References living entities versus non-living objects.
    Quantitative vs. Qualitative: Shows numerical data and measurements versus descriptive, non-numerical language.
    Generic vs. Specific: Assesses the use of general, non-specific language versus detailed and precise language.
    Social vs. Individual: Focuses on group dynamics and collective experiences versus personal experiences and individual perspectives.
    Positive vs. Negative: Reveals the presence of positive or negative sentiment and language.
    Short-term vs. Long-term: Expresses references to immediate or near-future timeframes versus distant future or enduring states.
    Formality vs. Informality: Evaluates the use of formal language and structures versus informal, colloquial language.
    Politeness: Shows the use of polite expressions and strategies.
    Directness: Evaluates the clarity and straightforwardness of communication.
    Novelty vs. Commonality: Assesses if new ideas and concepts are introduced versus references to familiar or widely known concepts.
    Intentionality: Evaluates the expression of intent or purpose.
    Objectivity: Measures the degree of impartiality and factual reporting.

System Architecture

The system uses two methods for analysis:

    Standard Methods: Utilize traditional, computationally efficient machine learning techniques and linguistic approaches (e.g., NLTK, TF-IDF, Bag of Words, rule-based methods).
    Advanced Methods: Incorporate free versions of large language models (LLMs) for sophisticated and in-depth analysis (e.g., GPT-2, GPT-Neo, DistilBERT).

The tool dynamically adapts to available computational resources, selecting the appropriate method to balance accuracy and efficiency.
Installation
Requirements

    Operating System: Windows, macOS, or Linux
    Python: Version 3.7 or higher
    pip: Python package installer
    Git (optional): For cloning the repository

Steps

    Clone the Repository

    Using Git:
    bash

git clone https://github.com/ovladon/Multidimensional_Text_Analysis.git

Or download the ZIP file from the repository page and extract it to your desired location.

Navigate to the Project Directory
bash

cd Multidimensional_Text_Analysis

Create a Virtual Environment (Optional but Recommended)

Create a virtual environment to manage dependencies:
bash

python -m venv venv

Activate the virtual environment:

    On Windows:
    bash

venv\Scripts\activate

On macOS/Linux:
bash

    source venv/bin/activate

Install Dependencies

Install the required Python packages using pip:
bash

pip install -r requirements.txt

Ensure that you have pip updated to the latest version:
bash

    pip install --upgrade pip

Usage

    Run the Application
    bash

    python3 start.py

    Ensure that start.py and the app.py is the main application file. Adjust the command if your main file has a different name.

    Input Text Data
        Use the interface to select text files or enter URLs.
        The tool supports processing multiple files simultaneously.

    Choose Analysis Method
        Standard Mode: Suitable for computers with limited resources.
        High-Performance Mode: Uses advanced LLMs for deeper analysis (requires more computational power).

    Select Dimensions
        Customize which dimensions you want to analyze.
        Add new dimensions as needed.

    View Results
        After analysis, view the results in the generated radar charts and comprehensive reports.
        Reports include scores for each dimension, identified synergies, and interpretations.

    Save or Export Results
        Download the reports and charts for further use or sharing.

Visualization

The tool provides visual representations of the analysis using radar charts:

    Single File Analysis: Displays the dimensions and their scores for one text.
    Multiple Files Comparison: Compares multiple texts on the same radar chart for easy comparison.
