from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from preprocessing import preprocess_reviews


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "datasets" / "IMDB_Dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print("Loading dataset...")

    df = pd.read_csv(DATASET_PATH)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    
    # Sentiment distribution
    

    df["sentiment"].value_counts().plot(kind="bar")

    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "sentiment_distribution.png",
        dpi=300
    )

    plt.close()

    
    # Review length
    

    df["review_length"] = df["review"].astype(str).apply(
        lambda x: len(x.split())
    )

    plt.hist(df["review_length"], bins=50)

    plt.title("Review Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "review_length_distribution.png",
        dpi=300
    )

    plt.close()

    
    # Preprocess reviews
    

    print("\nPreprocessing reviews for WordCloud...")

    df["cleaned_review"] = preprocess_reviews(df["review"])

    
    # Positive WordCloud
    

    positive_text = " ".join(
        df.loc[df["sentiment"] == "pos", "cleaned_review"]
    )

    positive_wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        max_words=100
    ).generate(positive_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(positive_wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Positive Reviews - Most Common Words")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "wordcloud_positive.png",
        dpi=300
    )

    plt.close()

    
    # Negative WordCloud


    negative_text = " ".join(
        df.loc[df["sentiment"] == "neg", "cleaned_review"]
    )

    negative_wordcloud = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        max_words=100
    ).generate(negative_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(negative_wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Negative Reviews - Most Common Words")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "wordcloud_negative.png",
        dpi=300
    )

    plt.close()

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()