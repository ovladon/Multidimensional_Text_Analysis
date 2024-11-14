# preprocessing.py

import io
from docx import Document
from pdfminer.high_level import extract_text
import pandas as pd

def preprocess_text(uploaded_file):
    """
    Preprocesses the uploaded file and extracts text.
    
    :param uploaded_file: The uploaded file.
    :return: Extracted text as a string.
    """
    try:
        if uploaded_file.type == "text/plain":
            return uploaded_file.read().decode('utf-8')
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Handle .docx files
            return extract_docx(uploaded_file)
        elif uploaded_file.type == "application/pdf":
            # Handle PDF files
            return extract_pdf(uploaded_file)
        elif uploaded_file.type == "text/csv":
            # Handle CSV files
            return extract_csv(uploaded_file)
        else:
            return ""
    except Exception as e:
        return ""

def extract_docx(uploaded_file):
    """
    Extracts text from a .docx file.
    
    :param uploaded_file: The uploaded .docx file.
    :return: Extracted text as a string.
    """
    try:
        doc = Document(io.BytesIO(uploaded_file.read()))
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        return ""

def extract_pdf(uploaded_file):
    """
    Extracts text from a PDF file.
    
    :param uploaded_file: The uploaded PDF file.
    :return: Extracted text as a string.
    """
    try:
        return extract_text(io.BytesIO(uploaded_file.read()))
    except Exception as e:
        return ""

def extract_csv(uploaded_file):
    """
    Extracts text from a CSV file by concatenating all text entries.
    
    :param uploaded_file: The uploaded CSV file.
    :return: Extracted text as a string.
    """
    try:
        df = pd.read_csv(uploaded_file)
        return ' '.join(df.astype(str).values.flatten())
    except Exception as e:
        return ""

