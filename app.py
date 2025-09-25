# ============================================
# 📌 Fake vs Real Detection using TF-IDF + Logistic Regression
# ============================================

import streamlit as st
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ============================
# Load spaCy model
# ============================
nlp = spacy.load("en_core_web_sm")

# ============================
# Preprocessing
# ============================
def preprocess(text):
    """Lemmatize + remove stopwords/punct using spaCy"""
    doc = nlp(str(text).lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct
    ]
    return " ".join(tokens)

# ============================
# Training function
# ============================
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    vectorizer = TfidfVectorizer(max_features=5000)  # limit for efficiency
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=500)  # solver auto chooses method
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, vectorizer, acc, report

# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="Fake vs Real Detection (TF-IDF + Logistic Regression)", layout="wide")
st.title("📰 Fake vs Real News Detection")
st.write("Using **TF-IDF features + Logistic Regression**")

uploaded_file = st.file_uploader("Upload politifact_full.csv", type=["csv"])

trained_model = None
trained_vectorizer = None

if uploaded_file:
    df = pd.read_csv(uploaded_file).head(5000)  # limit for speed
    st.write("Dataset Preview:", df.head())

    # Create binary target if not present
    if "BinaryTarget" not in df.columns:
        df["BinaryTarget"] = df["Rating"].apply(
            lambda x: 1 if x in ["true", "mostly-true", "half-true"] else 0
        )

    X = df["Statement"].apply(preprocess)
    y = df["BinaryTarget"]

    if st.button("Run Training"):
        trained_model, trained_vectorizer, acc, report = train_model(X, y)
        st.success(f"✅ Accuracy: {acc:.4f}")
        st.json(report)

    # ============================
    # Prediction on user input
    # ============================
    st.subheader("🔍 Test on Custom Input")
    user_text = st.text_area("Enter a statement to classify:")
    if st.button("Classify Statement") and user_text.strip():
        if trained_model and trained_vectorizer:
            processed = preprocess(user_text)
            vec = trained_vectorizer.transform([processed])
            pred = trained_model.predict(vec)[0]
            st.write("Prediction:", "✅ True" if pred == 1 else "❌ Fake")
        else:
            st.warning("⚠️ Please train the model first!")
