%%writefile app2.py
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import fitz  # PyMuPDF
from fpdf import FPDF
import base64

# Set API Key for Groq
os.environ["GROQ_API_KEY"] = "YOUR_API_KEY"
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# 🧠 LLM Setup
llm = ChatGroq(model="llama3-8b-8192", temperature=0.5)

# 🔹 Resume Summary Prompt
summary_prompt = PromptTemplate(
    input_variables=["resume_text"],
    template="""
You are an AI recruiter assistant. Summarize the following resume text into a concise profile:
- Key skills
- Years of experience
- Education
- Notable achievements

Resume:
{resume_text}
"""
)

# 🔹 Job Role Suggestion Prompt
role_prompt = PromptTemplate(
    input_variables=["resume_text"],
    template="""
Based on the resume below, suggest 3 suitable job roles (with job titles only, no explanation):

{resume_text}
"""
)

summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
role_chain = LLMChain(llm=llm, prompt=role_prompt)

# 📄 PDF Download Utility
def generate_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.cell(200, 10, txt=line.encode('latin-1', 'replace').decode('latin-1'), ln=True)
    pdf_file = "summary.pdf"
    pdf.output(pdf_file)
    return pdf_file

# 📤 PDF Downloader as Base64 link
def get_pdf_download_link(pdf_path):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="AI_Summary.pdf">📥 Download Summary as PDF</a>'
    return href

# 🌐 Streamlit App UI
st.set_page_config(page_title="AI Resume Summarizer", layout="wide")
st.title("📄 AI Resume Summarizer Agent")
st.markdown("Upload a resume and get recruiter-friendly summary, job roles & more!")

uploaded_file = st.file_uploader("📁 Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    resume_text = "\n".join([page.get_text() for page in doc])

    st.markdown("🧾 **Extracted Resume Text:**")
    st.text_area("Resume Content", resume_text, height=300)

    if st.button("🚀 Summarize Resume"):
        with st.spinner("Generating summary..."):
            summary = summary_chain.run(resume_text=resume_text)
            st.success("✅ Recruiter Summary")
            st.markdown(summary)

            # PDF download
            pdf_path = generate_pdf(summary)
            st.markdown(get_pdf_download_link(pdf_path), unsafe_allow_html=True)

            # Rating
            st.markdown("⭐ **Rate the Summary (1-5):**")
            rating = st.slider("Rating", 1, 5, 4)
            st.write(f"Your rating: {rating}/5")

            # Job Role Suggestion
            st.markdown("💼 **Suggested Job Roles:**")
            roles = role_chain.run(resume_text=resume_text)
            st.markdown(roles)
