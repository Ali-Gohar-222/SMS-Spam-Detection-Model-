# 📌 IMPORT LIBRARIES
import pandas as pd
import string
import nltk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 📌 DOWNLOAD NLTK RESOURCES
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# 📌 CLEANING FUNCTION
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return ' '.join(words)

# 📌 CHECK IF MODEL EXISTS (avoid retraining)
import os
if os.path.exists('spam_model.pkl') and os.path.exists('vectorizer.pkl'):
    print("📦 Loading saved model and vectorizer...")
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
else:
    # 📌 LOAD DATA
    df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
    df['clean_message'] = df['message'].apply(clean_text)

    # 📌 SPLIT
    X_train, X_test, y_train, y_test = train_test_split(df['clean_message'], df['label'], test_size=0.2, random_state=42)

    # 📌 VECTORIZE
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 📌 TRAIN
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # 📌 EVALUATE
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='spam')
    rec = recall_score(y_test, y_pred, pos_label='spam')
    print(f"✅ Trained model | Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['HAM','SPAM'], yticklabels=['HAM','SPAM'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # 📌 SAVE MODEL + VECTORIZER
    joblib.dump(model, 'spam_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("💾 Model and vectorizer saved as spam_model.pkl / vectorizer.pkl")

# 📌 INTERACTIVE PREDICTION LOOP
print("\n📲 SMS Spam Detector Ready!")
print("Type your message below (or type 'exit' to quit):")

while True:
    user_input = input("Your SMS: ")
    if user_input.lower() == 'exit':
        print("👋 Exiting. Thanks for testing!")
        break
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]
    print(f"🔎 Prediction: {prediction.upper()}")
    print(f"📊 Probability - HAM: {proba[0]:.3f}, SPAM: {proba[1]:.3f}\n")
