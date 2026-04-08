import streamlit as st
import joblib
import time

st.set_page_config(page_title="SMS Spam Detector", layout="centered")

st.title("Spam Detector")
st.write("Type a message and click **Predict** to classify it.")

@st.cache_resource
def load_assets():
    model = joblib.load("src/projects/Classification/SpamClassifier Project/model_nb.pkl")
    cv = joblib.load("src/projects/Classification/SpamClassifier Project/vectorizer.pkl")
    return model, cv

model, cv = load_assets()

# Text box
user_input = st.text_area("Enter your message:", height=150)

# Buttom
if st.button("Predict"):
    if user_input.strip():
        
        # Loading spinner
        with st.spinner("Analyzing..."):
            time.sleep(1)  

            # Vectorization
            input_vectorized = cv.transform([user_input])

            # Predict
            prediction = model.predict(input_vectorized)

        # Result
        if prediction[0] == 0:
            st.success("✅ This is **HAM (Not Spam)**")
        else:
            st.error("🚨 This is **SPAM**")

    else:
        st.warning("Please enter some text.")