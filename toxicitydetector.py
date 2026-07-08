import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import re
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
st.set_page_config(
    page_title="Toxic Comment Detection",
    page_icon="💬",
    layout="wide"
)

@st.cache_resource
def load_lstm_model():
    model = load_model("toxicity_model.keras")
    return model

model = load_lstm_model()

@st.cache_resource
def load_tokenizer():

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return tokenizer

tokenizer = load_tokenizer()

@st.cache_data
def load_data():

    train = pd.read_csv("train.csv")

    return train

train = load_data()

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# ===========================================
# Parameters
# ===========================================

MAX_LENGTH = 150

# ===========================================
# Text Cleaning
# ===========================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"[^\w\s]", "", text)

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

# ===========================================
# Prediction Function
# ===========================================

def predict_comment(comment):

    comment = clean_text(comment)

    seq = tokenizer.texts_to_sequences([comment])

    pad = pad_sequences(
        seq,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(pad, verbose=0)[0]

    return prediction

# ===========================================
# Sidebar
# ===========================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Page",

    [
        "🏠 Home",
        "📊 Data Insights",
        "💬 Single Prediction",
        "📁 Bulk Prediction",
        "📈 Model Performance"
    ]
)

# ===========================================
# HOME PAGE
# ===========================================

if page == "🏠 Home":

    st.title("💬 Toxic Comment Detection")

    st.markdown("---")

    st.write("""
This application predicts whether a comment contains toxic content using a **Deep Learning LSTM model**.

The model predicts the following six classes:

- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

The model was trained using the **Jigsaw Toxic Comment Classification** dataset.
""")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dataset",
            f"{len(train):,} Comments"
        )

    with col2:
        st.metric(
            "Model",
            "LSTM"
        )

    with col3:
        st.metric(
            "Output Labels",
            "6"
        )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(train.head())

    st.success("Use the sidebar to navigate through the application.")
    
    # ===========================================
# DATA INSIGHTS PAGE
# ===========================================

elif page == "📊 Data Insights":

    st.title("📊 Data Insights")

    st.markdown("---")

    # =============================
    # Basic Information
    # =============================

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", train.shape[0])

    with col2:
        st.metric("Columns", train.shape[1])

    with col3:
        st.metric("Missing Values", int(train.isnull().sum().sum()))

    st.markdown("---")

    # =============================
    # Dataset Preview
    # =============================

    st.subheader("Dataset Preview")

    st.dataframe(train.head())

    st.markdown("---")

    # =============================
    # Label Distribution
    # =============================

    st.subheader("Distribution of Toxicity Labels")

    label_counts = train[labels].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(
        label_counts.index,
        label_counts.values
    )

    ax.set_xlabel("Labels")
    ax.set_ylabel("Count")
    ax.set_title("Toxic Comment Distribution")

    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.markdown("---")

    # =============================
    # Comment Length Distribution
    # =============================

    st.subheader("Comment Length Distribution")

    comment_length = train["comment_text"].astype(str).apply(len)

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(
        comment_length,
        bins=40
    )

    ax.set_xlabel("Comment Length")

    ax.set_ylabel("Frequency")

    ax.set_title("Distribution of Comment Length")

    st.pyplot(fig)

    st.markdown("---")

    # =============================
    # Dataset Statistics
    # =============================

    st.subheader("Summary Statistics")

    stats = pd.DataFrame({

        "Metric":[
            "Average Length",
            "Maximum Length",
            "Minimum Length",
            "Duplicate Comments"
        ],

        "Value":[

            round(comment_length.mean(),2),

            comment_length.max(),

            comment_length.min(),

            train.duplicated().sum()

        ]

    })

    st.table(stats)

    st.markdown("---")

    # =============================
    # Label Counts Table
    # =============================

    st.subheader("Label Counts")

    label_df = pd.DataFrame({

        "Label":labels,

        "Count":[
            train["toxic"].sum(),
            train["severe_toxic"].sum(),
            train["obscene"].sum(),
            train["threat"].sum(),
            train["insult"].sum(),
            train["identity_hate"].sum()
        ]

    })

    st.dataframe(label_df)

    st.success("Dataset insights generated successfully.")
    
    # ===========================================
# SINGLE COMMENT PREDICTION
# ===========================================

elif page == "💬 Single Prediction":

    st.title("💬 Toxic Comment Prediction")

    st.markdown(
        "Enter a comment below to predict its toxicity across the six categories."
    )

    st.markdown("---")

    user_input = st.text_area(
        "Enter Comment",
        height=180,
        placeholder="Type or paste a comment here..."
    )

    if st.button("Predict"):

        if user_input.strip() == "":

            st.warning("Please enter a comment.")

        else:

            with st.spinner("Predicting..."):

                prediction = predict_comment(user_input)

            st.success("Prediction Completed")

            st.markdown("---")

            st.subheader("Prediction Scores")

            result_df = pd.DataFrame({
                "Label": labels,
                "Probability": prediction
            })

            result_df["Percentage"] = (
                result_df["Probability"] * 100
            ).round(2)

            # Display progress bars
            for label, score in zip(labels, prediction):

                st.write(f"**{label.replace('_',' ').title()}**")

                st.progress(float(score))

                st.write(f"{score*100:.2f}%")

            st.markdown("---")

            st.subheader("Prediction Table")

            st.dataframe(result_df)

            st.markdown("---")

            # Overall prediction
            if np.max(prediction) >= 0.5:

                predicted_label = labels[np.argmax(prediction)]

                st.error(
                    f"⚠ Toxic Comment Detected\n\n"
                    f"Predicted Class: **{predicted_label.replace('_',' ').title()}**"
                )

            else:

                st.success("✅ Non-Toxic Comment")

            st.markdown("---")

            st.subheader("Sample Test Comments")

            st.info(
                """
Positive Comment

I really appreciate your help. Thank you for your support.

Negative Comment

You are an idiot and I hate you.
                """
            )
            
# ===========================================
# BULK CSV PREDICTION
# ===========================================

elif page == "📁 Bulk Prediction":

    st.title("📁 Bulk Toxic Comment Prediction")

    st.write(
        "Upload a CSV file containing a **comment_text** column."
    )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")

        st.dataframe(df.head())

        if "comment_text" not in df.columns:

            st.error(
                "CSV must contain a column named 'comment_text'."
            )

        else:

            if st.button("Predict All Comments"):

                with st.spinner("Generating predictions..."):

                    predictions = []

                    for comment in df["comment_text"]:

                        pred = predict_comment(str(comment))

                        predictions.append(pred)

                prediction_df = pd.DataFrame(
                    predictions,
                    columns=labels
                )

                # Convert probabilities to percentages
                for col in labels:
                    prediction_df[col] = (
                        prediction_df[col] * 100
                    ).round(2)

                # Overall predicted label
                prediction_df["Predicted_Label"] = (
                    prediction_df[labels]
                    .idxmax(axis=1)
                )

                prediction_df["Confidence (%)"] = (
                    prediction_df[labels]
                    .max(axis=1)
                ).round(2)

                result = pd.concat(
                    [
                        df,
                        prediction_df
                    ],
                    axis=1
                )

                st.success("Prediction Completed")

                st.markdown("---")

                st.subheader("Prediction Results")

                st.dataframe(result)

                csv = result.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name="toxicity_predictions.csv",
                    mime="text/csv"
                )

                st.markdown("---")

                st.subheader("Prediction Summary")

                summary = (
                    result["Predicted_Label"]
                    .value_counts()
                    .reset_index()
                )

                summary.columns = [
                    "Label",
                    "Count"
                ]

                st.dataframe(summary)

                fig, ax = plt.subplots(figsize=(7,4))

                ax.bar(
                    summary["Label"],
                    summary["Count"]
                )

                ax.set_title("Predicted Toxicity Labels")

                ax.set_xlabel("Label")

                ax.set_ylabel("Count")

                plt.xticks(rotation=45)

                st.pyplot(fig)
                
# ===========================================
# MODEL PERFORMANCE
# ===========================================

elif page == "📈 Model Performance":

    st.title("📈 Model Performance")

    st.markdown("---")

    st.subheader("Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
**Algorithm Used**

• Deep Learning

• LSTM (Long Short-Term Memory)

• TensorFlow / Keras
        """)

    with col2:
        st.info("""
**Dataset**

• Jigsaw Toxic Comment Classification

• Multi-label Classification

• Six Output Classes
        """)

    st.markdown("---")

    st.subheader("Performance Metrics")

    # Replace these values with your actual results
    accuracy = 0.98
    loss = 0.05

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Validation Accuracy",
            f"{accuracy*100:.2f}%"
        )

    with c2:
        st.metric(
            "Validation Loss",
            f"{loss:.4f}"
        )

    st.markdown("---")

    st.subheader("Model Architecture")

    architecture = pd.DataFrame({

        "Layer":[
            "Embedding",
            "LSTM",
            "Dropout",
            "Dense",
            "Output"
        ],

        "Description":[
            "Word Embedding (128)",
            "64 Units",
            "0.5",
            "64 Neurons (ReLU)",
            "6 Neurons (Sigmoid)"
        ]

    })

    st.table(architecture)

    st.markdown("---")

    st.subheader("Output Labels")

    label_table = pd.DataFrame({
        "Class": labels
    })

    st.table(label_table)

    st.markdown("---")

    st.subheader("Project Workflow")

    st.write("""
1. Load Dataset

2. Clean Text

3. Tokenization

4. Padding

5. LSTM Model

6. Toxicity Prediction

7. Streamlit Deployment
    """)

    st.markdown("---")

    st.success("Model deployed successfully using Streamlit.")
    
# ===========================================
# FOOTER
# ===========================================

st.markdown("---")

st.caption(
    "Developed using ❤️ Streamlit, TensorFlow, and Python"
)