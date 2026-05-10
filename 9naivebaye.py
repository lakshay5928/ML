import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Updated Dataset URL
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"

# Load dataset
df = pd.read_csv(url, sep='\t', names=['label', 'message'])

print("\nDataset Loaded Successfully\n")
print(df.head())

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Model
model = MultinomialNB(alpha=1.0)
model.fit(X_train_vec, y_train)

# Prediction
y_pred = model.predict(X_test_vec)

print("\n======= MODEL EVALUATION =======\n")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Custom Prediction Function
def predict_message(msg):
    msg_vec = vectorizer.transform([msg])
    pred = model.predict(msg_vec)[0]
    return "SPAM" if pred == 1 else "HAM"

print("\n======= CUSTOM TEST =======\n")

test_msg = "Congratulations! You have won a free lottery ticket. Call now!"

print("Message:", test_msg)
print("Prediction:", predict_message(test_msg))