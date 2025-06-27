# app/resume_matcher.py

import streamlit as st
import fitz  # PyMuPDF
import docx2txt
import re
import spacy
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import sys

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ✅ Load JD dictionary (adjust path if needed)
with open("data/jd_dict.json") as f:
    jd_dict = json.load(f)

# Predefined skill/education keywords
skills_list = ['python', 'sql', 'tableau', 'power bi', 'machine learning']
edu_keywords = ['bachelor', 'bsc', 'msc', 'phd', 'computer science']

# Extract resume text from file
def extract_text(file):
    if file.type == "application/pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "".join([page.get_text() for page in doc])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    return ""

# Resume parsing function
def parse_resume(text):
    doc = nlp(text)
    return {
        'name': doc.ents[0].text if doc.ents else None,
        'email': re.search(r'[\w.-]+@[\w.-]+', text).group(0) if re.search(r'[\w.-]+@[\w.-]+', text) else None,
        'phone': re.search(r'\+?\d[\d\s\-]{8,}\d', text).group(0) if re.search(r'\+?\d[\d\s\-]{8,}\d', text) else None,
        'skills': list({s for s in skills_list if s in text.lower()}),
        'education': list({e for e in edu_keywords if e in text.lower()}),
        'experience': max([int(x) for x in re.findall(r'(\d+)\+?\s+years?', text.lower())], default=None)
    }

# Match score calculator
def compute_match_score(resume_text, jd_text):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([jd_text.lower(), resume_text.lower()])
    return round(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100, 2)

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Resume Matcher App", layout="centered")
st.title("📄 Resume Matcher Web App")
st.write("Upload your resume and get matched to a Job Description!")

# Select category
category = st.selectbox("Select Job Category", list(jd_dict.keys()))

# File uploader
uploaded_file = st.file_uploader("Upload Resume (.pdf or .docx)", type=["pdf", "docx"])

if uploaded_file and category:
    resume_text = extract_text(uploaded_file)
    parsed = parse_resume(resume_text)
    jd_text = jd_dict.get(category, "")

    # Compute score using skills only
    score = compute_match_score(" ".join(parsed["skills"]), jd_text)

    # Display results
    st.subheader("✅ Match Score")
    st.success(f"{score}% match to **{category}** JD")

    st.subheader("📋 Resume Details")
    st.markdown(f"**Name:** {parsed['name'] or 'N/A'}")
    st.markdown(f"**Email:** {parsed['email'] or 'N/A'}")
    st.markdown(f"**Phone:** {parsed['phone'] or 'N/A'}")
    st.markdown(f"**Skills:** {', '.join(parsed['skills']) or 'N/A'}")
    st.markdown(f"**Education:** {', '.join(parsed['education']) or 'N/A'}")
    st.markdown(f"**Experience:** {parsed['experience']} years" if parsed['experience'] is not None else "**Experience:** Not mentioned")

