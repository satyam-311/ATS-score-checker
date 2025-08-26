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
    return cosine_similarity(vecs[0:1], vecs[1:2])[0][0]


# -----------------------------
# 3. Batch-safe BERT Encoding
# -----------------------------
def batch_encode(texts, model, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Use convert_to_tensor=False to avoid warnings
        emb = model.encode(batch, convert_to_tensor=False)
        embeddings.extend(emb)
    return embeddings


# -----------------------------
# 4. Training Function
# -----------------------------
def train_model(dataset_path="resume_job_matching_dataset.csv",
                model_path="ats_model.pkl",
                tfidf_path="tfidf_vectorizer.pkl"):
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

    # Compute TF-IDF similarity
    df['tfidf_score'] = df.apply(
        lambda x: compute_tfidf_similarity(x['resume_clean'], x['jd_clean'], tfidf), axis=1
    )

    # Load SBERT
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Compute BERT embeddings in batches
    resumes = df['resume_clean'].tolist()
    jds = df['jd_clean'].tolist()
    emb_resumes = batch_encode(resumes, sbert_model, batch_size=32)
    emb_jds = batch_encode(jds, sbert_model, batch_size=32)

    # Compute BERT similarity
    df['bert_score'] = [util.cos_sim(r, j).item() for r, j in zip(emb_resumes, emb_jds)]

    # -----------------------------
    # Train-Test Split
    # -----------------------------
    X = df[['tfidf_score', 'bert_score']]
    y = df['match_score']
    y = y - y.min()  # shift labels for XGBoost

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost
    model = XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Save Model & TF-IDF
    joblib.dump(model, model_path)
    joblib.dump(tfidf, tfidf_path)
    print(f"\n✅ Model saved as {model_path} and TF-IDF as {tfidf_path}")


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    train_model()


