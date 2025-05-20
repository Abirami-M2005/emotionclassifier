import streamlit as st
import joblib

# Load models and tools using Streamlit's caching mechanism
@st.cache_resource
def load_models():
    clf = joblib.load("decision_tree_model.pkl")  # or 'logistic_regression_model.pkl'
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return clf, vectorizer, label_encoder

# Load the models
clf, vectorizer, label_encoder = load_models()

# App UI
st.title("Tweet Emotion Classifier")
st.write("Enter a tweet below and see what emotion it expresses!")

# Text input
user_input = st.text_area("Tweet Text")

# Predict button
if st.button("Predict Emotion"):
    if not user_input.strip():
        st.warning("Please enter a tweet.")
    else:
        X_input = vectorizer.transform([user_input])
        prediction = clf.predict(X_input)
        predicted_emotion = label_encoder.inverse_transform(prediction)[0]
        st.success(f"Predicted Emotion: **{predicted_emotion}**")
