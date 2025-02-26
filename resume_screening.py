import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    try:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"  # Preserve spacing between pages
        return text if text else None  # Return None if no text found
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

# Function to rank resumes based on the job description
def rank_resumes(job_description, resumes):
    # Combine the job description with resumes
    documents = [job_description] + resumes
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    # Calculate cosine similarity
    job_description_vector = vectors[0]  # First vector is the job description
    resume_vectors = vectors[1:]  # Remaining are resumes
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Streamlit App
st.title("AI Resume Screening & Candidate Ranking System")

# Job Description Input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# File Uploader for Resumes
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description.strip():
    st.header("Ranking Resumes")
    
    resumes = []
    filenames = []
    
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        if text:
            resumes.append(text)
            filenames.append(file.name)
        else:
            st.warning(f"No readable text found in {file.name}. Skipping this file.")
    
    if resumes:
        # Rank resumes
        scores = rank_resumes(job_description, resumes)

        # Create and sort the results DataFrame
        results = pd.DataFrame({"Resume": filenames, "Score": scores})
        results = results.sort_values(by="Score", ascending=False)

        # Display results
        st.write(results)
    else:
        st.warning("No valid resumes found. Please check the uploaded files.")
else:
    st.info("Please enter a job description and upload at least one resume to start.")
