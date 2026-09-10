from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_reviews


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "datasets" / "IMDB_Dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

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

    
    # Same random_state and test_size as training.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    
    # Load trained model and vectorizer
    

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    
    # Transform test data
    

    X_test_tfidf = vectorizer.transform(X_test)

    
    # Predictions
    

    predictions = model.predict(X_test_tfidf)

    
    # Metrics
    

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

    report = classification_report(
        y_test,
        predictions,
        target_names=["Negative", "Positive"]
    )

    
    # Save accuracy

    with open(
        OUTPUT_DIR / "accuracy.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"Accuracy : {accuracy:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall   : {recall:.4f}\n"
            f"F1 Score : {f1:.4f}\n"
        )

    
    # Save classification report
    

    with open(
        OUTPUT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    
    # Confusion Matrix
    

    cm = confusion_matrix(
        y_test,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"]
    )

    display.plot()

    plt.title("Sentiment Analysis - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    
    # ROC-AUC
    

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            X_test_tfidf
        )[:, 1]

        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities
        )

        auc_score = roc_auc_score(
            y_test,
            probabilities
        )

        plt.figure()

        plt.plot(
            fpr,
            tpr,
            label=f"ROC-AUC = {auc_score:.4f}"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "roc_curve.png",
            dpi=300
        )

        plt.close()

        with open(
            OUTPUT_DIR / "accuracy.txt",
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                f"ROC-AUC  : {auc_score:.4f}\n"
            )

    
    # Print results


    print("\n==============================")
    print("EVALUATION RESULTS")
    print("==============================")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(report)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()