import streamlit as st
import joblib
import re
import nltk
import base64
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

st.set_page_config(
    page_title="Hate Speech Detection",
    page_icon="🚫",
    layout="centered"
)

nltk.download('stopwords')
stop_words = stopwords.words('english')
ps = PorterStemmer()

model = joblib.load("hate_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            display: flex;
            justify-content: center;
        }}
        .block-container {{
            padding-top: 15vh;
            padding-bottom: 0;
        }}
        .main-card {{
            background-color: rgba(255,255,255,0.95);
            padding: 18px;
            border-radius: 14px;
            max-width: 650px;
            width: 100%;
            box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        }}
        .result-card {{
            margin-top: 10px;
            padding: 12px;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 800;
            text-align: center;
        }}
        .hate {{
            background-color: #fdecea;
            color: #b71c1c;
            border-left: 4px solid #b71c1c;
            font-weight: 900;
        }}
        .normal {{
            background-color: #eef4ff;
            color: #0d47a1;
            border-left: 4px solid #0d47a1;
            font-weight: 900;
        }}
        .bold-label {{
            font-weight: 800;
            font-size: 18px;
            margin-bottom: 4px;
            color: #1a1a1a;
        }}
        .main-title {{
            font-weight: 900;
            font-size: 24px;
            text-align: center;
            margin: 0;
            color: #1a1a1a;
        }}
        .sub-title {{
            font-weight: 600;
            text-align: center;
            color: #555555;
            font-size: 15px;
            margin-top: 4px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("bg.png")

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

st.markdown(
    """
    <div class="main-card">
        <h3 class="main-title">
            🚫 Hate Speech Detection
        </h3>
        <p class="sub-title">
            NLP & Machine Learning Based Project
        </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<p class='bold-label'>✍️ Enter your text:</p>",
    unsafe_allow_html=True
)

user_input = st.text_area(
    "",
    height=85,
    placeholder="Type your sentence here..."
)

analyze = st.button("🔍 Analyze", use_container_width=True)

if analyze:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)
        if prediction[0] == 1:
            st.markdown(
                """
                <div class="result-card hate">
                    ⚠️ HATE SPEECH DETECTED
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="result-card normal">
                    ✅ NORMAL / NON-HATE SPEECH
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown("</div>", unsafe_allow_html=True)