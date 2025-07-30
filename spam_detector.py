# spam_detector.py

import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

MODEL_PATH = 'model/spam_classifier.pkl'

def train_model(data_path='spam_training_data.csv'):
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)

    X = df['text']
    y = df['label']

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('nb', MultinomialNB())
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    print("[✔] Spam detection model trained and saved.")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Spam model not found. Run train_model() first.")
    return joblib.load(MODEL_PATH)

def predict_spam(model, email_text):
    label = model.predict([email_text])[0]
    prob = model.predict_proba([email_text])[0].max()
    return label, prob

train_model('spam_training_data.csv')
