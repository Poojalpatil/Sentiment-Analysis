import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import preprocess

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("../datasets/IMDB_Dataset.csv")

# Preprocess
df["review"] = df["review"].apply(preprocess)

# Convert labels
df["sentiment"] = df["sentiment"].map({
    "pos":1,
    "neg":0
})

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("../models/sentiment_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

# -----------------------------
# Convert Text
# -----------------------------
X = vectorizer.transform(df["review"])
y = df["sentiment"]

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X)

# -----------------------------
# Evaluation
# -----------------------------
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

report = classification_report(y, y_pred)

cm = confusion_matrix(y, y_pred)

# -----------------------------
# Create outputs folder
# -----------------------------
os.makedirs("../outputs", exist_ok=True)

# -----------------------------
# Save Accuracy
# -----------------------------
with open("../outputs/accuracy.txt","w") as file:

    file.write(f"Accuracy : {accuracy:.4f}\n")
    file.write(f"Precision : {precision:.4f}\n")
    file.write(f"Recall : {recall:.4f}\n")
    file.write(f"F1 Score : {f1:.4f}\n")

# -----------------------------
# Save Report
# -----------------------------
with open("../outputs/classification_report.txt","w") as file:

    file.write(report)

# -----------------------------
# Save Confusion Matrix
# -----------------------------
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative","Positive"]
)

disp.plot()

plt.savefig("../outputs/confusion_matrix.png")

plt.close()

print("\nEvaluation Completed Successfully.")

print("\nAccuracy :",accuracy)

print("Precision :",precision)

print("Recall :",recall)

print("F1 Score :",f1)

print("\nFiles Saved in outputs folder.")