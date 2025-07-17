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
        st.markdown("---")
st.subheader("📥 Predict a Single Question")

with st.form("manual_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        topic_input = st.text_input("Topic (e.g. Loops)", max_chars=30)
    with col2:
        marks_input = st.number_input("Marks", min_value=1, max_value=20, value=5)

    submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            topic_encoded = encoder.transform([topic_input])[0]
            
            # Estimate Repetition Count (you can improve this logic)
            avg_rep = df[df['Topic'] == topic_input]['RepetitionCount'].mean()
            estimated_rep = int(avg_rep) if not pd.isna(avg_rep) else 1

            X_single = pd.DataFrame([[topic_encoded, marks_input, estimated_rep]],
                                    columns=['TopicEncoded', 'Marks', 'RepetitionCount'])
            prediction = model.predict(X_single)[0]
            proba = model.predict_proba(X_single)[0][prediction]

            if prediction == 1:
                st.success(f"✅ This question is likely to appear! (Confidence: {proba:.2f})")
            else:
                st.warning(f"❌ This question is unlikely to appear. (Confidence: {proba:.2f})")

            st.caption(f"Estimated repetition count used: {estimated_rep}")

        except ValueError:
            st.error("⚠️ Topic not recognized. Please enter a topic from the training data.")
