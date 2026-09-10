from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_reviews


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "datasets" / "IMDB_Dataset.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATASET_PATH)

    df = df.dropna(subset=["review", "sentiment"])

    print(f"Dataset size: {len(df)}")

    
    # Convert labels
    

    df["label"] = df["sentiment"].map({
        "pos": 1,
        "neg": 0
    })

    if df["label"].isnull().any():
        raise ValueError(
            "Unexpected sentiment labels found in dataset."
        )

    
    # Preprocessing


    print("Preprocessing reviews...")

    df["cleaned_review"] = preprocess_reviews(df["review"])

    X = df["cleaned_review"]
    y = df["label"]

    
    # Train/Test split
    

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    
    # TF-IDF
    

    print("Creating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    X_test_tfidf = vectorizer.transform(X_test)

    print(
        f"TF-IDF training matrix shape: "
        f"{X_train_tfidf.shape}"
    )

    
    # Logistic Regression
    

    print("Training Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train_tfidf, y_train)

    
    # Save model
    

    joblib.dump(
        model,
        MODEL_DIR / "sentiment_model.pkl"
    )

    joblib.dump(
        vectorizer,
        MODEL_DIR / "tfidf_vectorizer.pkl"
    )

    print("\nModel saved successfully.")

    print(
        f"Model: {MODEL_DIR / 'sentiment_model.pkl'}"
    )

    print(
        f"Vectorizer: {MODEL_DIR / 'tfidf_vectorizer.pkl'}"
    )

    print("\nTraining completed.")


if __name__ == "__main__":
    main()