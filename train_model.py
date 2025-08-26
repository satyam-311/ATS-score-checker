import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# -----------------------------
# 1. Text Preprocessing
# -----------------------------
def clean_text(text):
    """Clean text by lowercasing and removing non-alphabetic characters."""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


# -----------------------------
# 2. Feature Computation
# -----------------------------
def compute_tfidf_similarity(resume, jd, tfidf):
    vecs = tfidf.transform([resume, jd])
    return cosine_similarity(vecs[0], vecs[1])[0][0]


def compute_bert_similarity(resume, jd, model):
    emb1 = model.encode(resume, convert_to_tensor=True)
    emb2 = model.encode(jd, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()


# -----------------------------
# 3. Training Function
# -----------------------------
def train_model(dataset_path="resume_job_matching_dataset.csv", model_path="ats_model.pkl"):
    # Load Dataset
    df = pd.read_csv(dataset_path)
    df = df[['resume', 'job_description', 'match_score']].dropna()

    # Clean Text
    df['resume_clean'] = df['resume'].apply(clean_text)
    df['jd_clean'] = df['job_description'].apply(clean_text)

    # TF-IDF Setup
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    all_texts = df['resume_clean'].tolist() + df['jd_clean'].tolist()
    tfidf.fit(all_texts)

    # Compute TF-IDF Similarity
    df['tfidf_score'] = df.apply(
        lambda x: compute_tfidf_similarity(x['resume_clean'], x['jd_clean'], tfidf), axis=1
    )

    # Load SBERT
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Compute BERT Similarity
    df['bert_score'] = df.apply(
        lambda x: compute_bert_similarity(x['resume_clean'], x['jd_clean'], sbert_model), axis=1
    )

    # -----------------------------
    # Fix: Train-Test Split with label shift
    # -----------------------------
    X = df[['tfidf_score', 'bert_score']]
    y = df['match_score']

    # Shift labels to start from 0 (important for XGBoost)
    y = y - y.min()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost
    model = XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Save Model
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved as {model_path}")


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    train_model()

