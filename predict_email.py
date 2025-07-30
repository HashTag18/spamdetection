import joblib

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Get user input
subject = input("Enter email subject: ")
body = input("Enter email body: ")

# Combine and vectorize
text = subject + " " + body
X = vectorizer.transform([text])

# Predict
prediction = model.predict(X)[0]
print(f"\n[🔎] Prediction: {'📬 Not Spam' if prediction == 'ham' else '🚫 Spam'}")

