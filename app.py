import streamlit as st
import pandas as pd
import joblib
import re

# Load model and encoder
model = joblib.load("c_question_model.pkl")
encoder = joblib.load("topic_encoder.pkl")

st.title("📘 C Programming PYQ Predictor")
st.markdown("Upload a question set or enter a question manually to predict if it will appear in the upcoming exam.")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload Cleaned Question CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Uploaded Data Preview")
    st.write(df.head())

    if 'Topic' in df.columns:
        df['TopicEncoded'] = encoder.transform(df['Topic'])

        # ✅ Show all unique topics in a nice display box (sorted)
        all_topics = sorted(df['Topic'].dropna().unique())
        st.markdown("### 🧠 Model is trained on these topics:")
        st.success(", ".join(all_topics))

    else:
        st.error("❌ 'Topic' column is missing in the uploaded CSV.")

    # 📊 Batch prediction
    if all(col in df.columns for col in ['TopicEncoded', 'Marks', 'RepetitionCount']):
        X_new = df[['TopicEncoded', 'Marks', 'RepetitionCount']]
        predictions = model.predict(X_new)
        df['Prediction'] = predictions

        st.subheader("🎯 Predictions from File")
        st.write(df[['Topic', 'Marks', 'RepetitionCount', 'Prediction']])
        st.success("✅ 1 - Will Appear, 0 - Won't Appear")
    else:
        st.error("❌ Required columns missing: TopicEncoded, Marks, RepetitionCount.")
else:
    st.warning("⚠️ Please upload a question CSV to enable topic matching and prediction.")

# 🧠 Predict from full question text (manual)
st.markdown("---")
st.subheader("🧠 Predict Using Full Question Text")

with st.form("full_question_input"):
    question_text = st.text_area("Enter your full question", height=100)
    marks_input = st.number_input("Marks", min_value=1, max_value=20, value=5)
    submit_question = st.form_submit_button("Predict")

    if submit_question:
        if uploaded_file is None:
            st.error("❌ Please upload a CSV file first. The system uses it to match topics and estimate repetition.")
        else:
            matched_topic = None
            question_lower = question_text.lower()

            all_topics = df['Topic'].dropna().unique()

            # ✅ Case-insensitive topic matching with full word search
            for topic in all_topics:
                if re.search(rf"\\b{re.escape(topic.lower())}\\b", question_lower):
                    matched_topic = topic
                    break

            if matched_topic:
                try:
                    topic_encoded = encoder.transform([matched_topic])[0]
                    avg_rep = df[df['Topic'].str.lower() == matched_topic.lower()]['RepetitionCount'].mean()
                    estimated_rep = int(avg_rep) if not pd.isna(avg_rep) else 1

                    X_single = pd.DataFrame([[topic_encoded, marks_input, estimated_rep]],
                                            columns=['TopicEncoded', 'Marks', 'RepetitionCount'])
                    prediction = model.predict(X_single)[0]
                    proba = model.predict_proba(X_single)[0][prediction]

                    if prediction == 1:
                        st.success(f"✅ Likely to Appear (Topic: {matched_topic}, Confidence: {proba:.2f})")
                    else:
                        st.warning(f"❌ Unlikely to Appear (Topic: {matched_topic}, Confidence: {proba:.2f})")

                    st.caption(f"Estimated repetition count used: {estimated_rep}")
                except ValueError:
                    st.error("⚠️ Error encoding topic. Please check topic values.")
            else:
                st.error("⚠️ Could not identify the topic in your question. Try using known keywords like 'arrays', 'loops', 'structures', etc.")
