# Import required libraries
import re
import string
import nltk

# Download NLTK data (only first time)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Create stopword set
stop_words = set(stopwords.words('english'))

# Create lemmatizer object
lemmatizer = WordNetLemmatizer()


# -----------------------------
# Step 1: Clean Text
# -----------------------------
def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra spaces
    text = text.strip()

    return text


# -----------------------------
# Step 2: Tokenization
# -----------------------------
def tokenize_text(text):

    tokens = word_tokenize(text)

    return tokens


# -----------------------------
# Step 3: Remove Stopwords
# -----------------------------
def remove_stopwords(tokens):

    filtered_words = []

    for word in tokens:

        if word not in stop_words:

            filtered_words.append(word)

    return filtered_words


# -----------------------------
# Step 4: Lemmatization
# -----------------------------
def lemmatize_words(tokens):

    words = []

    for word in tokens:

        words.append(lemmatizer.lemmatize(word))

    return words


# -----------------------------
# Complete Preprocessing
# -----------------------------
def preprocess(text):

    # Cleaning
    text = clean_text(text)

    # Tokenization
    tokens = tokenize_text(text)

    # Stopword Removal
    tokens = remove_stopwords(tokens)

    # Lemmatization
    tokens = lemmatize_words(tokens)

    # Convert list back to sentence
    sentence = " ".join(tokens)

    return sentence


# -----------------------------
# Test Program
# -----------------------------
if __name__ == "__main__":

    review = input("Enter a movie review:\n")

    result = preprocess(review)

    print("\nProcessed Review:")
    print(result)