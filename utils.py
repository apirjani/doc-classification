import re

def preprocess_text(text):
    """
    Apply preprocessing steps to the input text.
    """
    # Remove non-word characters except for spaces
    text = re.sub(r'[^\w\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Remove email addresses
    text = re.sub(r'\b\S+@\S+\.\S+\b', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Collapse multiple spaces into a single space and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text