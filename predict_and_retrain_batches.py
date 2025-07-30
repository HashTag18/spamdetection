# predict_and_retrain_batches.py
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

ALL_EMAILS = "all.csv"
REVIEWED_FILE = "reviewed_emails.csv"
MODEL_FILE = "model/spam_model.pkl"
VEC_FILE = "model/vectorizer.pkl"

def load_model_and_vectorizer():
    if os.path.exists(MODEL_FILE) and os.path.exists(VEC_FILE):
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VEC_FILE)
        print("[✅] Model and vectorizer loaded.")
    else:
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        print("[⚠️] No existing model found. New model created.")
    return model, vectorizer

def save_reviewed_batch(batch_df, predictions):
    if not os.path.exists(REVIEWED_FILE):
        reviewed_df = pd.DataFrame()
    else:
        reviewed_df = pd.read_csv(REVIEWED_FILE)

    batch_df = batch_df.copy()
    batch_df["predicted_label"] = predictions
    reviewed_df = pd.concat([reviewed_df, batch_df], ignore_index=True)
    reviewed_df.to_csv(REVIEWED_FILE, index=False)
    print(f"[💾] Batch predictions saved to {REVIEWED_FILE}")

def retrain_model():
    print("[🔁] Retraining model...")
    df = pd.read_csv(REVIEWED_FILE)
    df = df[df["predicted_label"].notnull()]
    X = df["subject"].fillna('') + " " + df["text"].fillna('')
    y = df["predicted_label"].astype(str).str.strip()

    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    X_vec = vectorizer.fit_transform(X)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_vec, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VEC_FILE)
    print(f"[✅] Retrained and saved model to {MODEL_FILE}")

def predict_batches(batch_size=50):
    df = pd.read_csv(ALL_EMAILS)
    model, vectorizer = load_model_and_vectorizer()

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size].copy()
        X = batch["subject"].fillna('') + " " + batch["text"].fillna('')
        X_vec = vectorizer.transform(X)

        preds = model.predict(X_vec)
        save_reviewed_batch(batch, preds)

        if (i + batch_size) % 50 == 0:
            retrain_model()
            model, vectorizer = load_model_and_vectorizer()

if __name__ == "__main__":
    predict_batches()
