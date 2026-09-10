import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


STOP_WORDS = set(stopwords.words("english"))

# Preserve important negation words for sentiment analysis
NEGATION_WORDS = {"not", "no", "nor", "never"}
STOP_WORDS -= NEGATION_WORDS

LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Clean and normalize a movie review."""

    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Handle common contractions
    contractions = {
        "can't": "can not",
        "won't": "will not",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "isn't": "is not",
        "wasn't": "was not",
        "weren't": "were not",
        "aren't": "are not",
        "couldn't": "could not",
        "wouldn't": "would not",
        "shouldn't": "should not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
    }

    for contraction, replacement in contractions.items():
        text = text.replace(contraction, replacement)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    cleaned_tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]

    return " ".join(cleaned_tokens)


def preprocess_reviews(reviews):
    """Apply clean_text() to a collection of reviews."""
    return reviews.apply(clean_text)