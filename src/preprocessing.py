import re
from nltk import download
from nltk.tokenize import word_tokenize

try:
    word_tokenize("test")
except LookupError:
    download("punkt")


def tokenize(text):
    """Convert text into tokens."""
    return word_tokenize(str(text))


def prepare_text(text):
    """Prepare text in the same format used by the TF-IDF vectorizer."""
    tokens = tokenize(text)
    clean_text = " ".join(tokens)
    return clean_text