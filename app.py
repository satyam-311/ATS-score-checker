import streamlit as st
import PyPDF2
import re
import numpy as np
import tempfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    if uploaded_file is not None:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
        return text
    return ""

# Function to clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text

# Function to calculate ATS score using TF-IDF + Cosine Similarity (baseline)
def calculate_ats_score_tfidf(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity * 100, 2)

# Function to calculate ATS score using Sentence-BERT (semantic similarity)
def calculate_ats_score_bert(resume_text, jd_text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([resume_text, jd_text], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return round(similarity * 100, 2)

# Function to find missing keywords (based on TF-IDF vocab)
def find_missing_keywords(resume_text, jd_text):
    jd_words = set(jd_text.split()) - ENGLISH_STOP_WORDS
    resume_words = set(resume_text.split()) - ENGLISH_STOP_WORDS
    missing = jd_words - resume_words
    return list(missing)

# Function to create a downloadable report
def generate_report(score_tfidf, score_bert, missing_keywords):
    report_content = f"ATS Resume Evaluation Report\n\n"
    report_content += f"TF-IDF Match Score: {score_tfidf}%\n"
    report_content += f"Semantic (BERT) Match Score: {score_bert}%\n\n"
    if missing_keywords:
        report_content += "Missing Keywords: " + ", ".join(missing_keywords) + "\n\n"
    else:
        report_content += "Missing Keywords: None\n\n"

    if score_bert > 70:
        report_content += "Strong match! Your resume aligns well with the job description.\n"
    elif score_bert > 40:
        report_content += "Moderate match. Consider adding more relevant keywords.\n"
    else:
        report_content += "Weak match. Resume needs significant improvement.\n"

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    with open(temp_file.name, "w", encoding="utf-8") as f:
        f.write(report_content)
    return temp_file.name

# Streamlit UI
st.set_page_config(page_title="ATS Resume Expert (Improved Model)")
st.header("ATS Tracking System (Improved: TF-IDF + BERT)")

# Inputs
input_text = st.text_area("Job Description: ", key="input")
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])

if uploaded_file is not None:
    st.write("✅ PDF Uploaded Successfully")

submit = st.button("Calculate ATS Score")

if submit:
    if uploaded_file is not None and input_text.strip() != "":
        resume_text = extract_text_from_pdf(uploaded_file)
        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(input_text)

        score_tfidf = calculate_ats_score_tfidf(resume_clean, jd_clean)
        score_bert = calculate_ats_score_bert(resume_clean, jd_clean)
        missing_keywords = find_missing_keywords(resume_clean, jd_clean)

        st.subheader("Results:")
        st.write(f"📊 TF-IDF ATS Match Score: **{score_tfidf}%**")
        st.write(f"🤖 Semantic (BERT) Match Score: **{score_bert}%**")
        st.write("🔑 Missing Keywords:", ", ".join(missing_keywords) if missing_keywords else "None")

        if score_bert > 70:
            st.success("✅ Strong match! Your resume aligns well with the job description.")
        elif score_bert > 40:
            st.warning("⚠️ Moderate match. Consider adding more relevant keywords.")
        else:
            st.error("❌ Weak match. Resume needs significant improvement.")

        # Generate downloadable report
        report_file = generate_report(score_tfidf, score_bert, missing_keywords)
        with open(report_file, "rb") as f:
            st.download_button(
                label="📥 Download ATS Report",
                data=f,
                file_name="ATS_Report.txt",
                mime="text/plain"
            )

        os.remove(report_file)

    else:
        st.write("⚠️ Please upload resume and enter job description.")


