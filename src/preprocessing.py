import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Download required NLTK resources if missing
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


STOP_WORDS = set(stopwords.words("english"))

NEGATION_WORDS = {"not", "no", "nor", "never"}
STOP_WORDS -= NEGATION_WORDS

LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

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

    text = re.sub(r"\d+", " ", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    cleaned_tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]

    return " ".join(cleaned_tokens)


def preprocess_reviews(reviews):
    return reviews.apply(clean_text)