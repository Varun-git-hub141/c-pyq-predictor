import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("c_question_model.pkl")
encoder = joblib.load("topic_encoder.pkl")

st.title("📘 C Programming PYQ Predictor")
st.markdown("Upload question data to predict whether it will appear in the upcoming exam.")

# Upload CSV
uploaded_file = st.file_uploader("Upload Cleaned Question CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Display uploaded data
    st.subheader("🔍 Uploaded Data")
    st.write(df.head())

    # Encode topic
    if 'Topic' in df.columns:
        df['TopicEncoded'] = encoder.transform(df['Topic'])
    else:
        st.error("Topic column missing.")
    
    # Check other required columns
    if all(col in df.columns for col in ['TopicEncoded', 'Marks', 'RepetitionCount']):
        X_new = df[['TopicEncoded', 'Marks', 'RepetitionCount']]
        predictions = model.predict(X_new)
        df['Prediction'] = predictions

        st.subheader("🎯 Predictions")
        st.write(df[['Topic', 'Marks', 'RepetitionCount', 'Prediction']])

        st.success("✅ 1 - Will Appear, 0 - Won't Appear")
    else:
        st.error("Required columns missing: TopicEncoded, Marks, RepetitionCount.")