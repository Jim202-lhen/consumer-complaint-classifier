import streamlit as st
import joblib
import re


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Consumer Complaint Classifier",
    page_icon="✦",
    layout="centered"
)


# ============================================================
# PASTEL THEME
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .stApp {
        background: linear-gradient(
            135deg,
            #f7f1fb 0%,
            #fff4f6 50%,
            #eef7fb 100%
        );
    }

    .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }


    /* Header */
    .brand {
        text-align: center;
        color: #9a86b5;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 4px;
        margin-bottom: 12px;
    }

    .main-title {
        text-align: center;
        color: #4d4657;
        font-size: 42px;
        font-weight: 700;
        line-height: 1.15;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        color: #817987;
        font-size: 17px;
        margin-bottom: 30px;
    }


    /* Intro card */
    .intro-card {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid #e4d8ec;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 28px;
        box-shadow: 0 8px 25px rgba(130, 110, 145, 0.10);
    }

    .intro-title {
        color: #8e78aa;
        font-size: 19px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .intro-text {
        color: #68616f;
        font-size: 15px;
        line-height: 1.7;
    }


    /* Complaint heading */
    .complaint-title {
        color: #4d4657;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 8px;
    }


    /* Text area */
    textarea {
        background-color: #ffffff !important;
        color: #4d4657 !important;
        border-radius: 14px !important;
        border: 2px solid #e5dced !important;
    }

    textarea:focus {
        border-color: #bca8d2 !important;
        box-shadow: 0 0 0 2px rgba(188, 168, 210, 0.15) !important;
    }


    /* Button */
    .stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 13px;
        border: 1px solid #d6c4e2;

        background: linear-gradient(
            90deg,
            #c9b7df,
            #e8bdca
        );

        color: #514858;
        font-size: 16px;
        font-weight: 700;

        box-shadow:
            0 7px 18px rgba(155, 130, 170, 0.18);
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #bea8d7,
            #dfafbe
        );

        color: #403846;
    }


    /* Prediction */
    .prediction-card {
        background: linear-gradient(
            135deg,
            #eee6f8,
            #f9e8ed
        );

        border: 1px solid #d9c9e6;
        border-radius: 18px;
        padding: 25px;
        margin-top: 25px;
        text-align: center;

        box-shadow:
            0 8px 25px rgba(130, 110, 145, 0.12);
    }

    .prediction-label {
        color: #80768a;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }

    .prediction-result {
        color: #80669e;
        font-size: 29px;
        font-weight: 700;
    }


    /* Category cards */
    .category-card {
        padding: 14px 8px;
        border-radius: 13px;
        text-align: center;
        font-size: 14px;
        font-weight: 600;
        color: #5b5363 !important;
        margin-bottom: 10px;
    }

    .purple {
        background-color: #eee7f8;
        border: 1px solid #ddd0eb;
    }

    .pink {
        background-color: #f9e6ec;
        border: 1px solid #efd0da;
    }

    .blue {
        background-color: #e5f1f8;
        border: 1px solid #cfe3ee;
    }

    .green {
        background-color: #e8f4ed;
        border: 1px solid #d1e8da;
    }

    .yellow {
        background-color: #faf2dc;
        border: 1px solid #eddfb9;
    }

    .peach {
        background-color: #f9e9df;
        border: 1px solid #efd6c5;
    }


    /* Section title */
    .section-title {
        color: #4d4657 !important;
        font-size: 24px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 18px;
    }


    /* Model information card */
    .model-card {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid #e4d8ec;
        border-radius: 18px;
        padding: 22px 25px;
        margin-top: 10px;
        box-shadow: 0 8px 25px rgba(130, 110, 145, 0.10);
    }

    .model-title {
        color: #80669e !important;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    .model-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 9px 0;
        border-bottom: 1px solid #eee5f2;
        color: #5b5363 !important;
        font-size: 15px;
    }

    .model-row:last-child {
        border-bottom: none;
    }

    .model-row span {
        color: #817987 !important;
    }

    .model-row strong {
        color: #4d4657 !important;
        font-weight: 600;
    }


    /* Streamlit default text */
    .stApp h2,
    .stApp h3,
    .stApp p,
    .stApp label {
        color: #4d4657 !important;
    }


    /* Horizontal divider */
    .stApp hr {
        border-color: #d9c9e6 !important;
    }


    /* Footer */
    .footer {
        text-align: center;
        color: #98909f;
        font-size: 13px;
        line-height: 1.7;
        margin-top: 35px;
    }

    .footer-accent {
        color: #9a82b5;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL, VECTORIZER AND CATEGORIES
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "consumer_complaint_model.pkl"
    )

    vectorizer = joblib.load(
        "consumer_complaint_vectorizer.pkl"
    )

    categories = joblib.load(
        "complaint_categories.pkl"
    )

    return model, vectorizer, categories


model, vectorizer, categories = load_model()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove numbers and punctuation
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="brand">NLP • MACHINE LEARNING</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">Consumer Complaint<br>Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent classification of consumer financial complaints'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    """
    <div class="intro-card">
        <div class="intro-title">✦ How it works</div>
        <div class="intro-text">
            Enter a consumer financial complaint below.
            The system preprocesses the text, converts it into
            TF-IDF features, and uses a trained machine learning
            classifier to predict the most appropriate financial
            product category.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# COMPLAINT INPUT
# ============================================================

st.markdown(
    '<div class="complaint-title">Your Complaint</div>',
    unsafe_allow_html=True
)

complaint = st.text_area(
    "Complaint",
    placeholder=(
        "For example: I noticed an incorrect account "
        "on my credit report and I want it removed."
    ),
    height=190,
    label_visibility="collapsed"
)


# ============================================================
# CLASSIFICATION BUTTON
# ============================================================

if st.button("✦  Classify Complaint"):

    if complaint.strip() == "":

        st.warning(
            "Please enter a complaint before continuing."
        )

    else:

        # Clean complaint using the same preprocessing
        # used in the latest notebook
        cleaned_complaint = clean_text(
            complaint
        )

        # Convert complaint to TF-IDF features
        complaint_vector = vectorizer.transform(
            [cleaned_complaint]
        )

        # Predict category using final model
        predicted_category = model.predict(
            complaint_vector
        )[0]

        # Prediction card
        st.markdown(
            '<div class="prediction-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="prediction-label">'
            'Predicted Financial Product'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="prediction-result">'
            f'{predicted_category}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# SUPPORTED CATEGORIES
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">Supported Categories</div>',
    unsafe_allow_html=True
)


# Display categories saved from the latest notebook
category_styles = [
    "purple",
    "pink",
    "blue",
    "green",
    "yellow",
    "peach"
]

columns = st.columns(3)

for index, category in enumerate(categories):

    with columns[index % 3]:

        colour = category_styles[
            index % len(category_styles)
        ]

        st.markdown(
            f'<div class="category-card {colour}">'
            f'{category}'
            f'</div>',
            unsafe_allow_html=True
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="model-card">

        <div class="model-title">
            Model Information
        </div>

        <div class="model-row">
            <span>Final Model</span>
            <strong>Tuned Multinomial Naive Bayes</strong>
        </div>

        <div class="model-row">
            <span>TF-IDF</span>
            <strong>Unigrams + Bigrams</strong>
        </div>

        <div class="model-row">
            <span>Naive Bayes Alpha</span>
            <strong>0.1</strong>
        </div>

        <div class="model-row">
            <span>Accuracy</span>
            <strong>86.67%</strong>
        </div>

        <div class="model-row">
            <span>Macro F1</span>
            <strong>79.71%</strong>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="footer">'
    'Consumer Complaint Classification System'
    '<br>'
    '<span class="footer-accent">'
    'TF-IDF + Classical Machine Learning'
    '</span>'
    '<br>'
    'Developed for NLP Assignment'
    '</div>',
    unsafe_allow_html=True
)