import streamlit as st
import joblib

# Load the saved models and tools
@st.cache(allow_output_mutation=True)
def load_models():
    clf = joblib.load("decision_tree_model.pkl")  # or logistic_regression_model.pkl if you prefer
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return clf, vectorizer, label_encoder

clf, vectorizer, label_encoder = load_models()

# App title and description
st.title("Emotion Classification from Tweets")
st.write("Enter a tweet below, and the model will predict the emotion it expresses.")

# User input area
user_input = st.text_area("Enter Tweet Text")

if st.button("Predict Emotion"):
    if not user_input.strip():
        st.warning("Please enter some text to classify.")
    else:
        # Transform input using TF-IDF vectorizer
        X_input = vectorizer.transform([user_input])
        
        # Make prediction
        prediction = clf.predict(X_input)
        predicted_emotion = label_encoder.inverse_transform(prediction)[0]
        
        # Display result
        st.success(f"Predicted Emotion: **{predicted_emotion}**")
