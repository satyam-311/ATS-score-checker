import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer, util
import re
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import textstat
import plotly.graph_objects as go

# -----------------------------
# 0. Page Configuration (Must be first)
# -----------------------------
st.set_page_config(
    page_title="ATS Resume Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI enhancements
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold; 
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 1. Load Models
# -----------------------------
@st.cache_resource
def load_models():
    xgb_model = joblib.load("ats_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return xgb_model, tfidf, sbert_model

# Load models with a spinner for better UX
with st.spinner("Loading AI Models..."):
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
        return score
    except:
        return 0

def readability_label(score):
    if score >= 50:
        return "🟢 Simple"
    elif score >= 25:
        return "🟠 Standard"
    else:
        return "🔴 Technical"

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
# 5. Helper: Gauge Chart
# -----------------------------
def create_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "ATS Fit Score"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2E86C1"},
            'steps': [
                {'range': [0, 40], 'color': "#ffcccb"},
                {'range': [40, 70], 'color': "#fff4cc"},
                {'range': [70, 100], 'color': "#d4edda"}],
        }
    ))
    fig.update_layout(height=250, margin={'t': 20, 'b': 20, 'l': 20, 'r': 20})
    return fig

# -----------------------------
# 6. Streamlit UI
# -----------------------------

# Sidebar for Instructions
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910791.png", width=100)
    st.title("ATS Scanner")
    st.markdown("### How to use:")
    st.markdown("1. Upload your Resume (PDF).")
    st.markdown("2. Paste the Job Description.")
    st.markdown("3. Click **Predict Match**.")
    st.markdown("---")
    st.info("💡 **Pro Tip:** Ensure your resume is text-selectable, not an image scan.")

# Main Header
st.title("📄 Smart Resume Matcher")
st.markdown("Optimize your resume for Applicant Tracking Systems (ATS) with AI-powered analysis.")
st.divider()

# Input Section (Columns)
col_input1, col_input2 = st.columns(2)

resume_text = ""
uploaded_file = None

with col_input1:
    st.subheader("1️⃣ Upload Resume")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_file is not None:
        resume_text = extract_text_from_pdf(uploaded_file)
        with st.expander("👁️ View Extracted Resume Text"):
            st.text_area("", resume_text, height=150)
        st.success("PDF Loaded Successfully")

with col_input2:
    st.subheader("2️⃣ Job Description")
    jd_input = st.text_area("Paste JD here...", height=200)

# Predict Button centered
st.write("")
if st.button("🚀 Predict Match Score", type="primary"):
    if resume_text.strip() == "" or jd_input.strip() == "":
        st.error("⚠️ Please upload a Resume and enter a Job Description first.")
    else:
        with st.spinner('Analyzing keywords, semantics, and readability...'):
            resume_clean = clean_text(resume_text)
            jd_clean = clean_text(jd_input)

            # Calculation
            tfidf_score = compute_tfidf_similarity(resume_clean, jd_clean, tfidf)
            bert_score = compute_bert_similarity(resume_clean, jd_clean, sbert_model)
            keyword_score = compute_keyword_overlap(resume_clean, jd_clean)
            readability_score = compute_readability(resume_text)
            readable_label_text = readability_label(readability_score)

            match_score = xgb_model.predict([[tfidf_score, bert_score]])[0]

            # Final Score Logic
            sim_score = (0.4*tfidf_score + 0.6*bert_score) * 100
            xgb_score = ((match_score + 1)/5) * 100
            final_ats_score = round((sim_score + xgb_score)/2, 2)

        # -----------------------------
        # Results Dashboard
        # -----------------------------
        st.divider()
        st.subheader("📊 Analysis Results")

        # Top Row: Gauge and Prediction
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.plotly_chart(create_gauge_chart(final_ats_score), use_container_width=True)

        with res_col2:
            st.markdown("### Model Prediction")
            class_labels = {1: "Poor Match 😞", 2: "Moderate Match 😐", 3: "Good Match 🙂", 4: "Excellent Match 🚀"}
            prediction = class_labels.get(match_score, 'Unknown')
            
            # Colored Callout based on score
            if match_score >= 3:
                st.success(f"**Verdict:** {prediction}")
            else:
                st.warning(f"**Verdict:** {prediction}")
                
            st.markdown("---")
            st.markdown(f"**Readability Score:** {readability_score:.1f} ({readable_label_text})")
            st.caption("A higher readability score means the resume is easier to read.")

        # Second Row: Detailed Metrics
        st.subheader("🔍 Detailed Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)

        with m_col1:
            st.metric("TF-IDF Match", f"{tfidf_score:.2f}", delta="Keyword Frequency")
        with m_col2:
            st.metric("BERT Semantic", f"{bert_score:.2f}", delta="Contextual Meaning")
        with m_col3:
            st.metric("Keyword Overlap", f"{keyword_score*100:.1f}%", delta="Direct Match")

        # -----------------------------
        # Download Section
        # -----------------------------
        report_content = f"""
        ATS Resume Matching Report
        ---------------------------
        Final ATS Score: {final_ats_score}/100
        Verdict: {prediction}

        -- Scores --
        TF-IDF Similarity: {tfidf_score:.3f}
        BERT Similarity: {bert_score:.3f}
        Keyword Overlap: {keyword_score*100:.2f}%
        Readability: {readability_score:.2f} ({readable_label_text})
        """
        
        st.download_button(
            label="📥 Download Full Report",
            data=report_content,
            file_name="ats_report.txt",
            mime="text/plain"
        )
