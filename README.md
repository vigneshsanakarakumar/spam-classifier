# 📧 Spam Email Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-yellow)

A machine learning web app that classifies email/SMS messages as **spam or not spam** in real time.  
Built with TF-IDF vectorization and Multinomial Naive Bayes, deployed via Streamlit.

🔗 **[Live Demo →](https://your-app-name.streamlit.app)** ← replace after deploying

---

## 🖥️ What It Does

- Paste any email or SMS message and get an instant spam/not-spam prediction
- Shows **spam probability** as a percentage with a visual progress bar
- Highlights which words in your message matched the model's vocabulary
- Tracks prediction history and session statistics
- Works on mobile too

---

## 🧠 How It Works

```
Raw text input
    │
    ▼
Preprocessing
(lowercase → strip URLs → normalize numbers → remove punctuation)
    │
    ▼
TF-IDF Vectorizer (6,000 features, unigrams + bigrams)
    │
    ▼
Multinomial Naive Bayes Classifier
    │
    ▼
Spam probability + Label (Spam / Not Spam)
```

**Why Naive Bayes?**  
It's the standard baseline for text classification. Fast to train, easy to explain in interviews, and performs surprisingly well on spam detection — achieving ~98% accuracy on this dataset.

---

## 📊 Model Performance

Trained on the **SMS Spam Collection Dataset** (UCI Machine Learning Repository) — 5,574 messages, 13% spam.

| Metric    | Ham (Not Spam) | Spam  |
|-----------|---------------|-------|
| Precision | 99%           | 96%   |
| Recall    | 99%           | 95%   |
| F1-Score  | 99%           | 96%   |
| **Accuracy** | **98.3%** |       |

5-Fold CV Accuracy: **98.1% ± 0.3%**

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/vigneshsanakarakumar/spam-email-classifier.git
cd spam-email-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (downloads dataset automatically, takes ~30 seconds)
python train.py

# 4. Launch the app
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub → select this repo → set **Main file: `app.py`**
4. Add a startup command in **Advanced settings**:
   ```
   python train.py
   ```
5. Click **Deploy** — live in ~2 minutes

---

## 📁 Project Structure

```
spam-email-classifier/
├── app.py              # Streamlit web app
├── train.py            # Model training script
├── requirements.txt    # Dependencies
├── model/
│   ├── spam_model.pkl      # Trained Naive Bayes model (generated)
│   └── vectorizer.pkl      # TF-IDF vectorizer (generated)
└── data/
    └── SMSSpamCollection   # Dataset (auto-downloaded by train.py)
```

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Scikit-learn | TF-IDF + Naive Bayes |
| Streamlit | Web app framework |
| UCI SMS Dataset | Training data |

---

## 👤 Author

**Vignesh Sankarakumar**  
[LinkedIn](https://www.linkedin.com/in/vignesh-sankarakumar/) · [GitHub](https://github.com/vigneshsanakarakumar)

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
