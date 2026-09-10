from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from preprocessing import preprocess_reviews


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "datasets" / "IMDB_Dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATASET_PATH)

    df = df.dropna(subset=["review", "sentiment"])

    df["label"] = df["sentiment"].map({
        "pos": 1,
        "neg": 0
    })

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

    
    # TF-IDF

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    
    # Models
    

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Naive Bayes": MultinomialNB(),

        "Linear SVM": LinearSVC(
            random_state=42
        )
    }

    results = []

    best_model = None
    best_f1 = 0
    best_model_name = None

    
    # Train and compare


    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train_tfidf, y_train)

        predictions = model.predict(X_test_tfidf)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions
        )

        recall = recall_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions
        )

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    
    # Save comparison
    

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1 Score",
        ascending=False
    )

    results_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    
    # Save best model
    

    joblib.dump(
        best_model,
        BASE_DIR / "models" / "sentiment_model.pkl"
    )

    joblib.dump(
        vectorizer,
        BASE_DIR / "models" / "tfidf_vectorizer.pkl"
    )

    print("\n==============================")
    print("MODEL COMPARISON")
    

    print(results_df.to_string(index=False))

    print("\nBest Model:")
    print(best_model_name)

    print(f"Best F1 Score: {best_f1:.4f}")

    print("\nBest model saved successfully.")


if __name__ == "__main__":
    main()