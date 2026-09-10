from pathlib import Path

import joblib

from preprocessing import clean_text


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


def predict_sentiment(review: str):

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    cleaned_review = clean_text(review)

    review_vector = vectorizer.transform(
        [cleaned_review]
    )

    prediction = model.predict(
        review_vector
    )[0]

    # Logistic Regression has predict_proba.
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            review_vector
        )[0]

        confidence = max(probabilities) * 100

    else:

        confidence = None

    if prediction == 1:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return sentiment, confidence


def main():

    print("=" * 50)
    print("IMDb SENTIMENT ANALYZER")
    print("=" * 50)

    print("\nType 'exit' to close the program.")

    while True:

        review = input(
            "\nEnter movie review: "
        ).strip()

        if review.lower() == "exit":
            print("\nProgram closed.")
            break

        if not review:
            print("Please enter a review.")
            continue

        sentiment, confidence = predict_sentiment(
            review
        )

        print("\nPrediction")
        print("-" * 30)
        print(f"Sentiment: {sentiment}")

        if confidence is not None:
            print(
                f"Model Probability: "
                f"{confidence:.2f}%"
            )


if __name__ == "__main__":
    main()