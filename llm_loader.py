# llm_loader.py

from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import torch

def load_llm_model(dimension_name):
    """
    Returns an LLM model appropriate for the given dimension.
    Tries multiple models from specialized to generic.
    :param dimension_name: Name of the dimension.
    :return: A Hugging Face pipeline object or None if no model could be loaded.
    """
    model_choices = {
        'Affective': [
            ('text-classification', 'j-hartmann/emotion-english-distilroberta-base'),
            ('sentiment-analysis', 'distilbert-base-uncased-finetuned-sst-2-english'),
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        'Animate': [
            ('ner', 'dbmdz/bert-large-cased-finetuned-conll03-english', {'aggregation_strategy': 'simple'}),
            ('ner', 'dslim/bert-base-NER', {'aggregation_strategy': 'simple'})
        ],
        'Commonality': [
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        'Cognitive': [
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        'Directness': [
            ('text-classification', 'mrm8488/distilroberta-finetuned-directness'),
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        'Dynamic': [
            ('text-classification', 'mrm8488/distilroberta-finetuned-dynamic-static'),
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        'Formality': [
            ('text-classification', 'skorlm/formality-classifier-roberta-base'),
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        'Generic': [
            ('text-classification', 'Yaxin/bert-base-cased-finetuned-generic-specific'),
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        'Inanimate': [
            ('ner', 'dbmdz/bert-large-cased-finetuned-conll03-english', {'aggregation_strategy': 'simple'}),
            ('ner', 'dslim/bert-base-NER', {'aggregation_strategy': 'simple'}),
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        'Individual': [
            ('ner', 'dbmdz/bert-large-cased-finetuned-conll03-english', {'aggregation_strategy': 'simple'}),
            ('ner', 'dslim/bert-base-NER', {'aggregation_strategy': 'simple'})
        ],
        'Informality': [
            ('text-classification', 'skorlm/formality-classifier-roberta-base'),
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        'Intentionality': [
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        'LongTerm': [
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        'Negative': [
            ('sentiment-analysis', 'distilbert-base-uncased-finetuned-sst-2-english'),
            ('text-classification', 'nlptown/bert-base-multilingual-uncased-sentiment')
        ],
        'Novelty': [
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        'Objectivity': [
            ('text-classification', 'cardiffnlp/twitter-roberta-base-sentiment'),
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        'Place': [
            ('ner', 'dbmdz/bert-large-cased-finetuned-conll03-english', {'aggregation_strategy': 'simple'}),
            ('ner', 'dslim/bert-base-NER', {'aggregation_strategy': 'simple'})
        ],
        'Politeness': [
            ('text-classification', 'PrithivirajDamodaran/roberta-base-go-emotions'),
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        'Positive': [
            ('sentiment-analysis', 'distilbert-base-uncased-finetuned-sst-2-english')
        ],
        'Qualitative': [
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        'Quantitative': [
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        'ShortTerm': [
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        'Social': [
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        'Specific': [
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        'Static': [
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        'Time': [
            ('zero-shot-classification', 'facebook/bart-large-mnli')
        ],
        # Add other dimensions as needed
    }

    if dimension_name in model_choices:
        for choice in model_choices[dimension_name]:
            try:
                if len(choice) == 2:
                    task, model_name = choice
                    model = pipeline(task, model=model_name)
                elif len(choice) == 3:
                    task, model_name, kwargs = choice
                    model = pipeline(task, model=model_name, **kwargs)
                print(f"Using model '{model_name}' for dimension '{dimension_name}'.")
                return model
            except Exception as e:
                print(f"Failed to load model '{choice[1]}' for dimension '{dimension_name}': {e}")
        print(f"Could not load any model for dimension '{dimension_name}'. Falling back to standard method.")
        return None
    else:
        print(f"No models defined for dimension '{dimension_name}'. Falling back to standard method.")
        return None

def get_embedding(text, model):
    """
    Generates an embedding or output for the text using the selected model.

    :param text: The input text.
    :param model: The pre-loaded LLM model.
    :return: The output from the LLM model.
    """
    try:
        result = model(text[:512])  # Ensure input text is truncated for smaller LLMs
        return result
    except Exception as e:
        raise ValueError(f"Error in generating embedding for text: {e}")

def calculate_perplexity(text):
    """
    Calculates the perplexity of a given text using GPT-2.

    :param text: The input text.
    :return: Perplexity score.
    """
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        inputs = tokenizer(text, return_tensors='pt')

        # Calculate loss
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()

        return perplexity
    except Exception as e:
        raise ValueError(f"Error in calculating perplexity: {e}")

def perform_sentiment_analysis(text):
    """
    Performs sentiment analysis on the input text using a pre-trained model.

    :param text: The input text.
    :return: Sentiment analysis result.
    """
    try:
        # Use the Hugging Face pipeline for sentiment analysis
        sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        result = sentiment_pipeline(text[:512])  # Ensure text length limit is respected for model
        return result
    except Exception as e:
        raise ValueError(f"Error in performing sentiment analysis: {e}")

