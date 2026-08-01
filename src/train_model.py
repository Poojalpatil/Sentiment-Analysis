import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Import preprocess function
from preprocessing import preprocess

# -------------------------
# Load Dataset
# -------------------------

dataset_path = "../datasets/IMDB_dataset.csv"

df = pd.read_csv(dataset_path)

print("Dataset Loaded Successfully!")
print(df.head())

# -------------------------
# Preprocess Reviews
# -------------------------

print("\nPreprocessing reviews...")

df["review"] = df["review"].apply(preprocess)

print("Preprocessing Completed!")

# -------------------------
# Convert Labels
# -------------------------

df["sentiment"] = df["sentiment"].map({
    "pos": 1,
    "neg": 0
})

print(df["sentiment"].value_counts())

print(df["sentiment"].isnull().sum())

# -------------------------
# TF-IDF
# -------------------------

print("\nCreating TF-IDF Features...")

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["review"])

y = df["sentiment"]

print("TF-IDF Completed!")

# -------------------------
# Train-Test Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -------------------------
# Train Model
# -------------------------

print("\nTraining Model...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model Training Completed!")

# -------------------------
# Save Model
# -------------------------

os.makedirs("../models", exist_ok=True)

joblib.dump(model, "../models/sentiment_model.pkl")

joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")

print("\nModel Saved Successfully!")