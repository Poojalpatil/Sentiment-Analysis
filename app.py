from pathlib import Path
import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image


from src.preprocessing import clean_text



# CONFIGURATION


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
DATASET_PATH = BASE_DIR / "datasets" / "IMDB_Dataset.csv"

MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"


# PAGE CONFIGURATION


st.set_page_config(
    page_title="Sentiment Analysis ",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CUSTOM CSS


st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 0;
        }

        .subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 30px;
        }

        .metric-card {
            padding: 20px;
            border-radius: 12px;
            background-color: #f7f7f7;
            text-align: center;
            border: 1px solid #e5e5e5;
        }

        .positive {
            font-size: 28px;
            font-weight: 700;
        }

        .negative {
            font-size: 28px;
            font-weight: 700;
        }

        .info-box {
            padding: 15px;
            border-radius: 10px;
            background-color: #f5f7fa;
            border: 1px solid #ddd;
        }
    </style>
    """,
    unsafe_allow_html=True,
)



# LOAD MODEL


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None

    if not VECTORIZER_PATH.exists():
        return None, None

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


model, vectorizer = load_model()



# LOAD DATASET


@st.cache_data
def load_dataset():
    if not DATASET_PATH.exists():
        return None

    df = pd.read_csv(DATASET_PATH)

    return df


df = load_dataset()


# HELPER FUNCTIONS


def get_prediction(review):
    """
    Predict sentiment for a single review.
    """

    cleaned_review = clean_text(review)

    transformed_review = vectorizer.transform([cleaned_review])

    prediction = model.predict(transformed_review)[0]

    probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed_review)[0]
        probability = float(np.max(probabilities))

    if prediction == 1:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return sentiment, probability, cleaned_review


def predict_batch(dataframe):
    """
    Predict sentiment for multiple reviews.
    """

    result_df = dataframe.copy()

    result_df["cleaned_review"] = result_df["review"].astype(str).apply(
        clean_text
    )

    transformed_reviews = vectorizer.transform(
        result_df["cleaned_review"]
    )

    predictions = model.predict(transformed_reviews)

    result_df["sentiment"] = np.where(
        predictions == 1,
        "Positive",
        "Negative",
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed_reviews)

        result_df["confidence"] = np.max(
            probabilities,
            axis=1,
        )

        result_df["confidence"] = (
            result_df["confidence"] * 100
        ).round(2)

    return result_df


def display_image(filename, title):
    """
    Display an image from the outputs directory if available.
    """

    image_path = OUTPUT_DIR / filename

    if image_path.exists():
        image = Image.open(image_path)

        st.image(
            image,
            caption=title,
            use_container_width=True,
        )

    else:
        st.info(
            f"{filename} has not been generated yet."
        )


def load_model_comparison():
    """
    Load model comparison results.
    """

    path = OUTPUT_DIR / "model_comparison.csv"

    if path.exists():
        return pd.read_csv(path)

    return None



# SIDEBAR

st.sidebar.title("🎬 Sentiment ")

st.sidebar.markdown(
    """
     Navigation

    Use the menu below to explore the application.
    """
)

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📝 Review Analyzer",
        "📂 Batch Prediction",
        "📊 Model Performance",
        "🤖 Model Comparison",
        "📈 Dataset Analytics",
        
    ],
)



# CHECK MODEL


if model is None or vectorizer is None:

    st.error(
        """
        ⚠️ Trained model not found.

        Please train the model first using:

        python src/model_comparison.py
        """
    )

    st.stop()



# HOME


if page == "🏠 Home":

    st.markdown(
        '<p class="main-title">🎬 Sentiment Analysis NLP</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="subtitle">'
        "Interactive Movie Review Sentiment Classification"
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        This application uses Natural Language Processing
        and Machine Learning to classify movie reviews as
        Positive or Negative.
        """
    )

    st.divider()

    
    # PROJECT METRICS
    

    if df is not None:

        total_reviews = len(df)

        positive_reviews = (
            df["sentiment"]
            .value_counts()
            .get("pos", 0)
        )

        negative_reviews = (
            df["sentiment"]
            .value_counts()
            .get("neg", 0)
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Reviews",
                f"{total_reviews:,}",
            )

        with col2:
            st.metric(
                "Positive Reviews",
                f"{positive_reviews:,}",
            )

        with col3:
            st.metric(
                "Negative Reviews",
                f"{negative_reviews:,}",
            )

        with col4:
            st.metric(
                "Classes",
                "2",
            )

    st.divider()

    
    # QUICK ANALYSIS
    

    st.subheader("🚀 Quick Sentiment Analysis")

    review = st.text_area(
        "Enter a movie review",
        placeholder=(
            "Example: This movie was fantastic. "
            "The acting and story were excellent!"
        ),
        height=150,
    )

    if st.button(
        "🔍 Analyze Sentiment",
        type="primary",
    ):

        if not review.strip():

            st.warning(
                "Please enter a movie review."
            )

        else:

            sentiment, probability, cleaned = get_prediction(
                review
            )

            if sentiment == "Positive":

                st.success(
                    f"### 😊 {sentiment} Sentiment"
                )

            else:

                st.error(
                    f"### 😞 {sentiment} Sentiment"
                )

            if probability is not None:

                st.metric(
                    "Prediction Confidence",
                    f"{probability * 100:.2f}%",
                )



# REVIEW ANALYZER


elif page == "📝 Review Analyzer":

    st.title("📝 Single Review Analyzer")

    st.write(
        "Enter any movie review and let the trained NLP model "
        "classify its sentiment."
    )

    review = st.text_area(
        "Movie Review",
        placeholder="Type your movie review here...",
        height=200,
    )

    if st.button(
        "Analyze Review",
        type="primary",
    ):

        if not review.strip():

            st.warning(
                "Please enter a review first."
            )

        else:

            sentiment, probability, cleaned = get_prediction(
                review
            )

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Prediction")

                if sentiment == "Positive":

                    st.success(
                        f"😊 {sentiment}"
                    )

                else:

                    st.error(
                        f"😞 {sentiment}"
                    )

            with col2:

                st.subheader("Confidence")

                if probability is not None:

                    st.metric(
                        "Model Confidence",
                        f"{probability * 100:.2f}%",
                    )

                else:

                    st.info(
                        "Probability is not available "
                        "for this model."
                    )

            with st.expander(
                "🔎 View Preprocessed Text"
            ):

                st.write(cleaned)


# BATCH PREDICTION


elif page == "📂 Batch Prediction":

    st.title("📂 Batch CSV Prediction")

    st.write(
        "Upload a CSV file containing movie reviews and "
        "predict their sentiment automatically."
    )

    st.info(
        "Your CSV should contain a column named `review`."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(
                uploaded_file
            )

            st.subheader("Uploaded Data")

            st.dataframe(
                uploaded_df.head(10),
                use_container_width=True,
            )

            if "review" not in uploaded_df.columns:

                st.error(
                    "The CSV must contain a column named "
                    "`review`."
                )

            else:

                if st.button(
                    "🚀 Predict All Reviews",
                    type="primary",
                ):

                    with st.spinner(
                        "Analyzing reviews..."
                    ):

                        results = predict_batch(
                            uploaded_df
                        )

                    st.success(
                        "Prediction completed successfully!"
                    )

                    st.subheader(
                        "Prediction Results"
                    )

                    st.dataframe(
                        results,
                        use_container_width=True,
                    )

                    # Summary
                    st.subheader(
                        "📊 Prediction Summary"
                    )

                    positive_count = (
                        results["sentiment"]
                        .eq("Positive")
                        .sum()
                    )

                    negative_count = (
                        results["sentiment"]
                        .eq("Negative")
                        .sum()
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Total",
                            len(results),
                        )

                    with col2:

                        st.metric(
                            "Positive",
                            positive_count,
                        )

                    with col3:

                        st.metric(
                            "Negative",
                            negative_count,
                        )

                    # Download
                    csv_data = results.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="⬇️ Download Predictions",
                        data=csv_data,
                        file_name="sentiment_results.csv",
                        mime="text/csv",
                    )

        except Exception as e:

            st.error(
                f"Error processing CSV: {e}"
            )



# MODEL PERFORMANCE


elif page == "📊 Model Performance":

    st.title("📊 Model Performance")

    accuracy_file = OUTPUT_DIR / "accuracy.txt"
    report_file = OUTPUT_DIR / "classification_report.txt"

    
    # METRICS
    

    if report_file.exists():

        report_text = report_file.read_text(
            encoding="utf-8"
        )

        st.subheader(
            "Classification Report"
        )

        st.code(
            report_text,
            language="text",
        )

    else:

        st.warning(
            "Classification report not found."
        )

    st.divider()

    
    # ACCURACY
    

    if accuracy_file.exists():

        accuracy_text = accuracy_file.read_text(
            encoding="utf-8"
        )

        st.subheader(
            "Model Evaluation Results"
        )

        st.code(
            accuracy_text,
            language="text",
        )


    # CONFUSION MATRIX
    

    st.divider()

    st.subheader(
        "Confusion Matrix"
    )

    display_image(
        "confusion_matrix.png",
        "Confusion Matrix",
    )

    
    # ROC CURVE
    

    st.subheader(
        "ROC Curve"
    )

    display_image(
        "roc_curve.png",
        "ROC Curve",
    )



# MODEL COMPARISON


elif page == "🤖 Model Comparison":

    st.title("🤖 Machine Learning Model Comparison")

    st.write(
        "Compare the performance of different machine-learning "
        "classifiers used for sentiment analysis."
    )

    comparison_df = load_model_comparison()

    if comparison_df is None:

        st.warning(
            """
            Model comparison results are not available.

            Run:

            python src/model_comparison.py
            """
        )

    else:

        st.dataframe(
            comparison_df,
            use_container_width=True,
        )

        st.divider()

        
        # BAR CHART
        

        st.subheader(
            "📊 Performance Comparison"
        )

        numeric_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        ]

        available_metrics = [
            metric
            for metric in numeric_metrics
            if metric in comparison_df.columns
        ]

        if available_metrics:

            chart_df = comparison_df.set_index(
                comparison_df.columns[0]
            )[available_metrics]

            st.bar_chart(chart_df)

        
        # BEST MODEL
        

        if "f1_score" in comparison_df.columns:

            best_index = comparison_df[
                "f1_score"
            ].idxmax()

            best_model = comparison_df.loc[
                best_index
            ]

            st.success(
                f"""
                🏆 Best Model:

                {best_model.iloc[0]}

                F1-Score:
                {best_model['f1_score']:.4f}
                """
            )


# DATASET ANALYTICS


elif page == "📈 Dataset Analytics":

    st.title("📈 Dataset Analytics")

    if df is None:

        st.error(
            "IMDb dataset was not found."
        )

    else:

        
        # DATASET INFORMATION
    

        st.subheader(
            "Dataset Overview"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Rows",
                f"{len(df):,}",
            )

        with col2:

            st.metric(
                "Columns",
                len(df.columns),
            )

        with col3:

            missing_values = int(
                df.isnull().sum().sum()
            )

            st.metric(
                "Missing Values",
                missing_values,
            )

        

        with st.expander(
            "👀 View Dataset Sample"
        ):

            st.dataframe(
                df.head(20),
                use_container_width=True,
            )

        st.divider()

        
        # SENTIMENT DISTRIBUTION
        

        st.subheader(
            "😊 Sentiment Distribution"
        )

        sentiment_counts = (
            df["sentiment"]
            .value_counts()
        )

        chart_data = pd.DataFrame(
            {
                "Sentiment": sentiment_counts.index,
                "Reviews": sentiment_counts.values,
            }
        )

        st.bar_chart(
            chart_data.set_index(
                "Sentiment"
            )
        )

        st.divider()

        
        # REVIEW LENGTH
        

        st.subheader(
            "📏 Review Length Distribution"
        )

        review_lengths = (
            df["review"]
            .astype(str)
            .str.len()
        )

        length_df = pd.DataFrame(
            {
                "Review Length": review_lengths
            }
        )

        st.line_chart(
            length_df.head(500)
        )

        st.divider()

        
        # GENERATED VISUALIZATIONS
        

        st.subheader(
            "☁️ Word Clouds"
        )

        col1, col2 = st.columns(2)

        with col1:

            display_image(
                "wordcloud_positive.png",
                "Positive Reviews Word Cloud",
            )

        with col2:

            display_image(
                "wordcloud_negative.png",
                "Negative Reviews Word Cloud",
            )




    st.divider()

    st.subheader(
        "👨‍💻 Project Author"
    )

    st.write(
        "**Pooja Patil**"
    )

    st.write(
        "GitHub: github.com/Poojalpatil"
    )

    st.caption(
        "Sentiment Analysis NLP — Machine Learning Project"
    )



# FOOTER


st.sidebar.divider()

st.sidebar.caption(
    "🎬 Sentiment Analysis NLP"
)

st.sidebar.caption(
    "Built with Python, NLP, Scikit-learn & Streamlit"
)

