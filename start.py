import subprocess
import sys

def run_setup():
    """Run the environment setup script."""
    print("Running environment setup...")
    subprocess.check_call([sys.executable, 'environment.py'])

def run_streamlit():
    """Start the Streamlit application."""
    print("Starting the Streamlit app...")
    subprocess.check_call([sys.executable, '-m', 'streamlit', 'run', 'app.py'])

if __name__ == '__main__':
    run_setup()
    run_streamlit()

