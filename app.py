import gradio as gr
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data only if not already
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = joblib.load("best_model_linear_svm.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english') and len(t) > 2]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Prediction function
def predict_sentiment(review):
    processed = preprocess(review)
    features = vectorizer.transform([processed])
    prediction = model.predict(features)[0]
    return "✅ Positive 😊" if prediction == "positive" else "❌ Negative 😠"

# Gradio Interface
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, placeholder="Enter your movie review here..."),
    outputs=gr.Textbox(label="Prediction"),
    title="🎬 Movie Review Sentiment Analyzer",
    description="Enter a review to find out whether it's Positive or Negative."
)

interface.launch()
import zipfile

# Unzip and read CSV
with zipfile.ZipFile("processed_reviews.zip", "r") as zip_ref:
    zip_ref.extractall()

import pandas as pd
df = pd.read_csv("processed_reviews.csv")