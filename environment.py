import subprocess
import sys
import os
import nltk

def create_environment():
    """
    Installs necessary packages, models, and sets up the environment.
    Ensures everything is ready for the app to run smoothly across different operating systems.
    """
    print("Setting up the environment...")

    # Install necessary Python libraries
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    except Exception as e:
        print(f"Error during pip install: {e}")

    # Install SpaCy model
    try:
        print("Downloading SpaCy model...")
        import spacy
        spacy.cli.download('en_core_web_sm')
    except Exception as e:
        print(f"Error during SpaCy model installation: {e}")

    # Download necessary NLTK data
    try:
        required_packages = [
            'punkt',
            'averaged_perceptron_tagger',
            'wordnet',
            'stopwords',
            'brown',
            'opinion_lexicon',
            'universal_tagset',
            'tagsets'
        ]
        for package in required_packages:
            nltk.download(package)
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

    print("Environment setup complete.")

def detect_system_resources():
    """
    Detects available system resources (RAM, CPU, GPU) to choose the appropriate mode.
    Returns a dictionary with the resource details.
    """
    import psutil

    system_info = {}

    # Get system RAM size in GB
    system_info['ram'] = psutil.virtual_memory().total / (1024 ** 3)

    # Check CPU cores
    system_info['cpu'] = os.cpu_count()

    # Check for GPU availability (simple check using torch)
    try:
        import torch
        system_info['gpu'] = torch.cuda.is_available()
    except ImportError:
        system_info['gpu'] = False  # No GPU available

    return system_info

if __name__ == '__main__':
    create_environment()

