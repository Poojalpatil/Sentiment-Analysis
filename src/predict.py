import joblib

# Import preprocess function
from preprocessing import preprocess

# -------------------------
# Load Model
# -------------------------

print("Loading Model...")

model = joblib.load("../models/sentiment_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

print("Model Loaded Successfully!\n")

# -------------------------
# Prediction Loop
# -------------------------

while True:

    review = input("Enter Movie Review (or type 'exit' to quit):\n")

    if review.lower() == "exit":
        print("Program Closed.")
        break

    # Preprocess review
    processed_review = preprocess(review)

    # Convert text to TF-IDF
    review_vector = vectorizer.transform([processed_review])

    # Predict
    prediction = model.predict(review_vector)

    # Display result
    if prediction[0] == 1:
        print("\nPrediction : Positive 😊\n")
    else:
        print("\nPrediction : Negative 😞\n")