Python 3.12.1 (tags/v3.12.1:2305ca5, Dec  7 2023, 22:03:25) [MSC v.1937 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import streamlit as st
... import joblib
... 
... @st.cache_resource
... def load_models():
...     clf = joblib.load("decision_tree_model.pkl")  # or logistic_regression_model.pkl if preferred
...     vectorizer = joblib.load("tfidf_vectorizer.pkl")
...     label_encoder = joblib.load("label_encoder.pkl")
...     return clf, vectorizer, label_encoder
... 
... # Load models
... clf, vectorizer, label_encoder = load_models()
... 
... # Streamlit UI
... st.title("Tweet Emotion Classifier")
... st.write("Enter a tweet below and see what emotion it expresses!")
... 
... user_input = st.text_area("Tweet Text")
... 
... if st.button("Predict Emotion"):
...     if not user_input.strip():
...         st.warning("Please enter a tweet.")
...     else:
...         X_input = vectorizer.transform([user_input])
...         prediction = clf.predict(X_input)
...         predicted_emotion = label_encoder.inverse_transform(prediction)[0]
