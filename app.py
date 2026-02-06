import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from transformers import pipeline

from preprocess_review import preprocess


st.set_page_config(
    page_title="Sentiment Analyzer App",
    layout="wide"
)

@st.cache_resource
def load_ml_models():
    return {
        "Naive Bayes": joblib.load("best_models/naive_bayes.pkl"),
        "Logistic Regression": joblib.load("best_models/logistic_regression.pkl"),
        "Random Forest": joblib.load("best_models/random_forest.pkl")
    }

@st.cache_resource
def load_bert_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

ml_models = load_ml_models()
bert_model = load_bert_model()

# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("product_reviews.csv")
    df["clean_review"] = df["review_text"].astype(str).apply(preprocess)
    df["true_sentiment"] = df["sentiment"].map({1: "Positive", 0: "Negative"})
    return df

df = load_data()


# --------------------------------------------------
st.sidebar.title("⚙️ Controls")

section = st.sidebar.selectbox(
    "Select Section",
    [
        "🔍 Sentiment Prediction",
        "📉 Review Insights",
        "📊 Rating vs Sentiment Analytics"
    ]
)

model_choice = st.sidebar.radio(
    "Choose Model",
    ["Naive Bayes", "Logistic Regression", "Random Forest", "BERT (Deep Learning)"]
)

# --------------------------------------------------
SENTIMENT PREDICTION
# --------------------------------------------------
if section == "🔍 Sentiment Prediction":

    st.title("🛒 Product Review Sentiment Analyzer")

    review_text = st.text_area("✍️ Enter a product review", height=180)

    if st.button("Analyze Sentiment"):
        if review_text.strip() == "":
            st.warning("Please enter a review.")
        else:
            if model_choice == "BERT (Deep Learning)":
                result = bert_model(review_text)[0]
                label = result["label"].capitalize()
                confidence = result["score"]

            else:
                cleaned = preprocess(review_text)
                pred = ml_models[model_choice].predict([cleaned])[0]
                label = "Positive" if pred == 1 else "Negative"
                confidence = None

            if label.lower().startswith("pos"):
                st.success(f"✅ {label} Review")
            else:
                st.error(f"❌ {label} Review")

            if confidence:
                st.caption(f"Confidence: {confidence:.2f}")

# --------------------------------------------------
 REVIEW INSIGHTS
# --------------------------------------------------
elif section == "📉 Review Insights":

    st.title("📉 Negative Review Insights")

    neg_df = df[df["true_sentiment"] == "Negative"]

    all_words = " ".join(neg_df["clean_review"]).split()
    common_words = Counter(all_words).most_common(15)

    keywords_df = pd.DataFrame(
        common_words,
        columns=["Keyword", "Frequency"]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔑 Top Complaint Keywords")
        st.dataframe(keywords_df, use_container_width=True)

    with col2:
        st.subheader("☁️ WordCloud")

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(" ".join(neg_df["clean_review"]))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

# --------------------------------------------------
 RATING VS SENTIMENT ANALYTICS
# --------------------------------------------------
elif section == "📊 Rating vs Sentiment Analytics":

    st.title("📊 Sentiment Analytics Dashboard")

    # Predict sentiment using Logistic Regression (baseline)
    df["predicted_sentiment"] = ml_models["Logistic Regression"].predict(
        df["clean_review"]
    )
    df["predicted_sentiment"] = df["predicted_sentiment"].map(
        {1: "Positive", 0: "Negative"}
    )

    summary = (
        df.groupby(["true_sentiment", "predicted_sentiment"])
        .size()
        .reset_index(name="Count")
    )

    st.subheader("Confusion-style Summary")
    st.dataframe(summary, use_container_width=True)

    mismatch = df[df["true_sentiment"] != df["predicted_sentiment"]]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", len(df))
    col2.metric("Mismatch Reviews", len(mismatch))
    col3.metric(
        "Mismatch %",
        f"{(len(mismatch) / len(df)) * 100:.2f}%"
    )

    st.info(
        "Mismatch reviews reveal users whose ratings and textual sentiment "
        "do not align, highlighting hidden dissatisfaction or bias."
    )


st.markdown("---")

