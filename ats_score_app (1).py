import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer, util
import re
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import textstat

# -----------------------------
# 1. Load Models
# -----------------------------
@st.cache_resource
def load_models():
    xgb_model = joblib.load("ats_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return xgb_model, tfidf, sbert_model

xgb_model, tfidf, sbert_model = load_models()

# -----------------------------
# 2. Text Preprocessing
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# -----------------------------
# 3. Feature Computation
# -----------------------------
def compute_tfidf_similarity(resume, jd, tfidf):
    vecs = tfidf.transform([resume, jd])
    return cosine_similarity(vecs[0:1], vecs[1:2])[0][0]

def compute_bert_similarity(resume, jd, model):
    emb1 = model.encode(resume, convert_to_tensor=True)
    emb2 = model.encode(jd, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

def compute_keyword_overlap(resume, jd):
    resume_words = set(resume.split())
    jd_words = set(jd.split())
    if len(jd_words) == 0:
        return 0
    return len(resume_words & jd_words) / len(jd_words)

def compute_readability(text):
    try:
        score = textstat.flesch_reading_ease(text)
        return score  # raw score rakh lo, label ke liye use hoga
    except:
        return 0

def readability_label(score):
    if score >= 60:
        return "🟢 Easy"
    elif score >= 30:
        return "🟠 Moderate"
    else:
        return "🔴 Difficult"


# -----------------------------
# 4. PDF Text Extraction
# -----------------------------
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text.strip()

# -----------------------------
# 5. Streamlit UI
# -----------------------------
st.title("📄 ATS Resume Matcher")
st.write("Upload a resume (PDF) and enter Job Description to predict final ATS score with detailed metrics.")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
resume_text = ""
if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Resume Text Preview", resume_text, height=200)

jd_input = st.text_area("Job Description")

if st.button("Predict Match Score"):
    if resume_text.strip() == "" or jd_input.strip() == "":
        st.warning("Please upload Resume and enter Job Description.")
    else:
        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(jd_input)

        # -----------------------------
        # Calculate all metrics
        # -----------------------------
        tfidf_score = compute_tfidf_similarity(resume_clean, jd_clean, tfidf)
        bert_score = compute_bert_similarity(resume_clean, jd_clean, sbert_model)
        keyword_score = compute_keyword_overlap(resume_clean, jd_clean)
        readability_score = compute_readability(resume_text)
        readable_label = readability_label(readability_score)

        match_score = xgb_model.predict([[tfidf_score, bert_score]])[0]

        # Final ATS Score calculation
        sim_score = (0.4*tfidf_score + 0.6*bert_score) * 100
        xgb_score = ((match_score + 1)/5) * 100
        final_ats_score = round((sim_score + xgb_score)/2, 2)

        # -----------------------------
        # Display Metrics neatly
        # -----------------------------
        st.success(f"✅ Final ATS Score: {final_ats_score}/100")

        class_labels = {1: "Poor Match", 2: "Moderate Match", 3: "Good Match", 4: "Excellent Match"}

        col1, col2 = st.columns(2)
        with col1:
            st.metric("TF-IDF Similarity", f"{tfidf_score:.2f}", help="Keyword-based similarity")
            st.metric("Keyword Match", f"{keyword_score*100:.2f}%", help="JD keywords found")
        with col2:
            st.metric("Semantic (BERT) Similarity", f"{bert_score:.2f}", help="Contextual similarity")
            st.metric("Resume Readability", readable_label)

        st.info(f"📌 ATS Prediction: **{class_labels.get(match_score, 'Unknown')}**")

        # -----------------------------
        # Downloadable report
        # -----------------------------
        report_content = f"""
ATS Resume Matching Report
---------------------------

Final ATS Score: {final_ats_score}/100

Resume (Preview):
{resume_text[:1000]}...

Job Description:
{jd_input[:1000]}...

Similarity Scores:
TF-IDF: {tfidf_score:.3f}
BERT: {bert_score:.3f}
Keyword Match: {keyword_score*100:.2f}%
Readability: {readability_score:.2f} ({readable_label})

XGBoost Predicted Class: {class_labels.get(match_score, 'Unknown')}
"""
        report_bytes = report_content.encode('utf-8')
        st.download_button(
            label="📥 Download ATS Report",
            data=report_bytes,
            file_name="ats_report.txt",
            mime="text/plain"
        )
