import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer, util
import re
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

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
st.write("Upload a resume (PDF) and enter Job Description to predict final ATS score.")

# Resume upload
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
resume_text = ""
if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Resume Text Preview", resume_text, height=200)

# Job Description input
jd_input = st.text_area("Job Description")

# Prediction
if st.button("Predict Match Score"):
    if resume_text.strip() == "" or jd_input.strip() == "":
        st.warning("Please upload Resume and enter Job Description.")
    else:
        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(jd_input)

        tfidf_score = compute_tfidf_similarity(resume_clean, jd_clean, tfidf)
        bert_score = compute_bert_similarity(resume_clean, jd_clean, sbert_model)
        match_score = xgb_model.predict([[tfidf_score, bert_score]])[0]

        # Final ATS Score calculation (average of similarities + XGBoost class)
        tfidf_norm = tfidf_score
        bert_norm = bert_score
        sim_score = (0.4*tfidf_norm + 0.6*bert_norm) * 100
        xgb_score = ((match_score + 1)/5) * 100
        final_ats_score = round((sim_score + xgb_score)/2, 2)

        st.success(f"✅ Final ATS Score: {final_ats_score}/100")
        st.write(f"(TF-IDF: {tfidf_score:.3f}, BERT: {bert_score:.3f}, XGBoost class: {match_score})")

        # -----------------------------
        # 6. Create downloadable report
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

XGBoost Predicted Class: {match_score}
"""

        report_bytes = report_content.encode('utf-8')
        st.download_button(
            label="📥 Download ATS Report",
            data=report_bytes,
            file_name="ats_report.txt",
            mime="text/plain"
        )
