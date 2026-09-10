# Sentiment Analysis 

End-to-end NLP application for automated movie-review sentiment classification.

An interactive machine-learning application that analyzes movie reviews and classifies them as Positive or Negative using NLP preprocessing, TF-IDF feature engineering, and multiple machine-learning algorithms.

Built with Python, NLTK, Scikit-learn, and Streamlit.

# Highlights

* End-to-end NLP preprocessing pipeline
* TF-IDF with unigram and bigram features
* Comparison of Logistic Regression, Naive Bayes, and Linear SVM
* Evaluation using Accuracy, Precision, Recall, F1-Score, and ROC-AUC
* Interactive Streamlit dashboard
* Single-review sentiment prediction
* Batch CSV sentiment prediction
* Confusion matrix, ROC curve, and dataset analytics
* Trained model and TF-IDF vectorizer saved using **Joblib**



# Tech Stack

Language: Python 

NLP: NLTK

ML: Scikit-learn

Feature Engineering: TF-IDF

Data: Pandas, NumPy

Visualization: Matplotlib, WordCloud

Web App: Streamlit

Model Serialization: Joblib


## ⚙️ Run Locally

### 1. Clone


git clone https://github.com/Poojalpatil/Sentiment-Analysis-NLP.git
cd Sentiment-Analysis-NLP


### 2. Create Virtual Environment

python -m venv venv
.\venv\Scripts\Activate.ps1

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Train the Model

python src/model_comparison.py


### 5. Run the Application

streamlit run app.py
# Application

The Streamlit application provides:

Review Analyzer – Classify individual movie reviews.
Batch Prediction – Analyze multiple reviews from a CSV file.
Model Performance – View accuracy, precision, recall, F1-score, and other metrics.
Model Comparison – Compare different machine learning algorithms.
Dataset Analytics – Explore IMDb review data and visualizations.



# Dataset

IMDb Movie Reviews Dataset

Binary sentiment labels:

 pos → Positive
 neg → Negative

# Major Challenges & Solutions

* Noisy Review Text - Built an NLP preprocessing pipeline to clean HTML, URLs, punctuation, and stopwords.
* Sentiment Context & Negation - Preserved important words such as not, no, and never.
* Choosing the Best Model - Compared Logistic Regression, Naive Bayes, and Linear SVM using multiple evaluation metrics.
* Data Leakage - Fitted TF-IDF only on training data and evaluated on unseen test data.
* Model Deployment - Integrated the trained NLP model into a Streamlit application for real-time and batch predictions.



# Skills Demonstrated

Python , NLP , Text Preprocessing , TF-IDF , Machine Learning , Model Evaluation , Scikit-learn , Streamlit , Data Visualization.


