import streamlit as st
import pandas as pd
import pickle
import os
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


st.set_page_config(page_title="Plagiarism Checker AI")


file_path = "train_snli.txt"  # if you diffrent data set  Update this with the correct dataset by  Er.Anoop Gupta 
model_path = "plagiarism_model.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"


@st.cache_data
def load_data():
    try:
        #  this is initialization and Load dataset
        data = pd.read_csv(file_path, sep="\t", header=None, names=["Text1", "Text2", "Label"])
        
        
        def preprocess_text(text):
            if isinstance(text, float):
                return ""
            text = text.lower()  
            text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
            return text

        # Apply text cleaning Processing 
        data["Text1"] = data["Text1"].apply(preprocess_text)
        data["Text2"] = data["Text2"].apply(preprocess_text)

        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Load data
st.info("Loading dataset...")
data = load_data()

# Train model if not already saved
if data is not None:
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.info("Training model for the first time...")
        
        # Convert text into numerical format using TF-IDF
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(data["Text1"] + " " + data["Text2"])
        y = data["Label"]
        
        # Train model
        model = LogisticRegression(max_iter=500)
        model.fit(X, y)
        
        # Save trained model and vectorizer
        with open(model_path, "wb") as model_file:
            pickle.dump(model, model_file)
        with open(vectorizer_path, "wb") as vec_file:
            pickle.dump(tfidf, vec_file)
        
        st.success("Model trained and saved successfully!")


        #Developed by Er.Anoop Kumar Gupta 
        #Instagram Account :  me_anoopkumargupta
    
    # Load trained model
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_path, "rb") as vec_file:
        tfidf = pickle.load(vec_file)

    # Function to check plagiarism
    def check_plagiarism(text1, text2):
        text1, text2 = text1.lower(), text2.lower()
        input_features = tfidf.transform([text1 + " " + text2])
        prediction = model.predict(input_features)
        return "Plagiarized" if prediction[0] == 1 else "Not Plagiarized"

    #   This is Streamlit UI for taken as a input  we dot's need any UI for  html,css javascript.
    st.header("🔍 Plagiarism Checker AI")
    text1 = st.text_area("Enter First Text:")
    text2 = st.text_area("Enter Second Text:")
    
    if st.button("Check Plagiarism"):
        if text1 and text2:
            result = check_plagiarism(text1, text2)
            st.subheader("Result:")
            st.write(result)
        else:
            st.warning("Please enter both texts!")





            
