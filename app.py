import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# Streamlit UI
st.title("Sentiment Analysis: Decode Emotions in Text")
user_input = st.text_area("Enter a social media message:")

if st.button("Analyze"):
    clean = clean_text(user_input)
    vector = vectorizer.transform([clean])
    prediction = model.predict(vector)[0]

    if prediction == 0:
        st.error("Sentiment: Negative")
    elif prediction == 1:
        st.info("Sentiment: Neutral")
    else:
        st.success("Sentiment: Positive")
