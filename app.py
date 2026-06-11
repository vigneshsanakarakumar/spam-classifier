"""
app.py
------
Spam Email Classifier — Streamlit Web App
Run: streamlit run app.py
"""

import streamlit as st
import pickle
import os
import re
from pathlib import Path

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 750px; }

    .result-spam {
        background: #FFF0F0;
        border-left: 5px solid #E03131;
        padding: 18px 22px;
        border-radius: 6px;
        margin: 16px 0;
    }
    .result-ham {
        background: #F0FFF4;
        border-left: 5px solid #2F9E44;
        padding: 18px 22px;
        border-radius: 6px;
        margin: 16px 0;
    }
    .label-spam { color: #E03131; font-size: 1.6rem; font-weight: 700; }
    .label-ham  { color: #2F9E44; font-size: 1.6rem; font-weight: 700; }

    .confidence-bar-wrap {
        background: #eee;
        border-radius: 20px;
        height: 12px;
        margin: 8px 0 4px 0;
        overflow: hidden;
    }
    .stat-box {
        background: #F8F9FA;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 6px 0;
        font-size: 0.95rem;
    }
    .history-item {
        padding: 8px 12px;
        border-radius: 6px;
        margin: 4px 0;
        font-size: 0.88rem;
    }
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = Path("spam_model.pkl")
    vectorizer_path = Path("vectorizer.pkl")

    if not model_path.exists() or not vectorizer_path.exists():
        return None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def preprocess(text):
    """Basic text cleaning."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_top_words(text, vectorizer, n=8):
    """Return top spam-signal words found in the input."""
    cleaned = preprocess(text)
    words = cleaned.split()
    vocab = set(vectorizer.vocabulary_.keys())
    found = [w for w in words if w in vocab]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for w in found:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique[:n]


def classify(text, model, vectorizer):
    """Run prediction and return label, confidence."""
    cleaned = preprocess(text)
    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0]
    label = model.predict(vec)[0]
    confidence = proba[1] if label == "spam" else proba[0]
    spam_prob = proba[1]
    return label, confidence, spam_prob


# ── Session State ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "count_spam" not in st.session_state:
    st.session_state.count_spam = 0
if "count_ham" not in st.session_state:
    st.session_state.count_ham = 0


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("📧 Spam Email Classifier")
st.caption("Built with Naive Bayes + TF-IDF · Trained on the SMS Spam Collection dataset")
st.markdown("---")

model, vectorizer = load_model()

if model is None:
    st.error("⚠️  Model not found. Run `python train.py` first to train the model.")
    st.code("python train.py", language="bash")
    st.stop()

# ── Input ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Paste your email or message")

user_input = st.text_area(
    label="Input",
    placeholder="Type or paste any email/SMS message here...",
    height=160,
    label_visibility="collapsed"
)

# ── Example buttons ───────────────────────────────────────────────────────────
st.caption("Try an example:")
ex_col1, ex_col2, ex_col3 = st.columns(3)

spam_example = "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim your prize now: www.win-prize.com"
ham_example  = "Hey, are we still on for the meeting tomorrow at 3pm? Let me know if you need to reschedule."
phish_example = "URGENT: Your bank account has been suspended. Verify your details immediately at secure-login.net"

if ex_col1.button("🚨 Spam example"):
    user_input = spam_example
if ex_col2.button("✅ Normal example"):
    user_input = ham_example
if ex_col3.button("⚠️ Phishing example"):
    user_input = phish_example

st.markdown("")

# ── Classify ──────────────────────────────────────────────────────────────────
if st.button("🔍  Classify Message", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter a message to classify.")
    else:
        label, confidence, spam_prob = classify(user_input, model, vectorizer)
        top_words = get_top_words(user_input, vectorizer)

        is_spam = label == "spam"

        # ── Result card ───────────────────────────────────────
        if is_spam:
            st.markdown(f"""
            <div class="result-spam">
                <span class="label-spam">🚨 SPAM</span><br>
                <span style="color:#555; font-size:0.95rem;">This message looks like spam. Do not click any links.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-ham">
                <span class="label-ham">✅ NOT SPAM</span><br>
                <span style="color:#555; font-size:0.95rem;">This message appears to be legitimate.</span>
            </div>
            """, unsafe_allow_html=True)

        # ── Confidence ────────────────────────────────────────
        st.markdown(f"**Spam probability: {spam_prob:.1%}**")
        st.progress(spam_prob)

        # ── Signal words ──────────────────────────────────────
        if top_words:
            st.markdown("**Words matched in vocabulary:**")
            st.markdown(" ".join([f"`{w}`" for w in top_words]))

        # ── Stats ─────────────────────────────────────────────
        if is_spam:
            st.session_state.count_spam += 1
        else:
            st.session_state.count_ham += 1

        # History
        preview = user_input[:55] + "..." if len(user_input) > 55 else user_input
        st.session_state.history.insert(0, {
            "label": label,
            "preview": preview,
            "confidence": confidence,
            "spam_prob": spam_prob
        })
        st.session_state.history = st.session_state.history[:10]


# ── Session Stats ─────────────────────────────────────────────────────────────
total = st.session_state.count_spam + st.session_state.count_ham
if total > 0:
    st.markdown("---")
    st.subheader("📊 This Session")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Checked", total)
    c2.metric("🚨 Spam", st.session_state.count_spam)
    c3.metric("✅ Not Spam", st.session_state.count_ham)


# ── History ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.subheader("🕓 Recent Predictions")
    for item in st.session_state.history:
        color = "#FFF0F0" if item["label"] == "spam" else "#F0FFF4"
        icon  = "🚨" if item["label"] == "spam" else "✅"
        st.markdown(f"""
        <div class="history-item" style="background:{color};">
            {icon} <b>{item['label'].upper()}</b> ({item['spam_prob']:.0%} spam) — {item['preview']}
        </div>
        """, unsafe_allow_html=True)

    if st.button("Clear history"):
        st.session_state.history = []
        st.session_state.count_spam = 0
        st.session_state.count_ham = 0
        st.rerun()


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built by Vignesh Sankarakumar · 
    <a href="https://github.com/vigneshsanakarakumar/spam-email-classifier" target="_blank">GitHub</a> · 
    Trained on SMS Spam Collection Dataset (UCI)
</div>
""", unsafe_allow_html=True)
