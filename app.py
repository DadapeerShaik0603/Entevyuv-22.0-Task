import streamlit as st
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import re

# Download necessary NLTK data
nltk.download('punkt')

# Define the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the summarization pipeline
summarizer = pipeline("summarization")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    # Remove citations
    text = re.sub(r'\[\d+\]', '', text)
    # Remove multi-column layout artifacts
    text = re.sub(r'\n+', '\n', text)
    sentences = sent_tokenize(text)
    cleaned_sentences = [sentence.lower() for sentence in sentences]
    return cleaned_sentences

# Function to chunk text
def chunk_text(sentences, chunk_size=500):
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# Function to create embeddings
def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings

# Function to store embeddings in FAISS index
def store_embeddings(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return index

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, index, chunks, k=5):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding).astype('float32'), k)
    relevant_chunks = [chunks[i] for i in I[0]]
    return relevant_chunks

# Function to generate response
def generate_response(relevant_chunks):
    combined_text = " ".join(relevant_chunks)
    summary = summarizer(combined_text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Function to add metadata
def add_metadata(chunks, text):
    metadata_chunks = []
    sections = ["Abstract", "Introduction", "Methodology", "Results and Discussion", "Conclusion"]
    for chunk in chunks:
        section = None
        for sec in sections:
            if sec in chunk:
                section = sec
                break
        metadata_chunks.append({"text": chunk, "section": section})
    return metadata_chunks

# Streamlit app
st.title("RAG Application for AI Research Papers")

uploaded_file = st.file_uploader("Upload a research paper", type="pdf")
if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("uploaded_paper.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text from the uploaded PDF
    text = extract_text_from_pdf("uploaded_paper.pdf")
    cleaned_text = preprocess_text(text)
    chunks = chunk_text(cleaned_text)
    metadata_chunks = add_metadata(chunks, text)
    embeddings = create_embeddings([chunk['text'] for chunk in metadata_chunks])
    index = store_embeddings(embeddings)

    query = st.text_input("Enter your query")
    if query:
        relevant_chunks = retrieve_relevant_chunks(query, index, [chunk['text'] for chunk in metadata_chunks])
        response = generate_response(relevant_chunks)
        st.write("Response:", response)
