import streamlit as st
import pdfplumber
import re
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
import faiss
import numpy as np
import time

load_dotenv()  # reads the .env file and loads GEMINI_API_KEY into the environment

st.set_page_config(page_title="Intelligent Document Assistant", layout="wide")


# ---------- Phase 1: Ingestion ----------
def extract_all_pages(pdf_path):
    all_pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            all_pages_text.append(text)
    return "\n".join(all_pages_text)


# ---------- Phase 2: Cleaning ----------
def clean_text(raw_text):
    text = raw_text
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ---------- Phase 2: Chunking ----------
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# ---------- Phase 3: Load embedding model (cached so it only loads once) ----------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


# ---------- Gemini client ----------
@st.cache_resource
def load_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


embedding_model = load_embedding_model()
client = load_gemini_client()


# ---------- Full pipeline: PDF -> chunks + FAISS index ----------
def build_index(pdf_path):
    raw_text = extract_all_pages(pdf_path)
    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned, chunk_size=500, overlap=150)
    embeddings = embedding_model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return chunks, index


# ---------- Phase 5: Retrieval + Generation ----------
def generate_answer(question, chunks, index, top_k=3, max_retries=3):
    question_embedding = embedding_model.encode([question])
    distances, indices = index.search(np.array(question_embedding), top_k)
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""You are a helpful assistant answering questions based only on the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return "Sorry, the model is currently unavailable. Please try again later."


# ---------- Streamlit UI ----------
st.title("📄 Intelligent Document Assistant")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    temp_path = os.path.join("data", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("Processing document... this may take a minute for large files."):
            chunks, index = build_index(temp_path)
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.current_file = uploaded_file.name
        st.success(f"Processed {len(st.session_state.chunks)} chunks from {uploaded_file.name}")

    question = st.text_input("Ask a question about the document:")

    if st.button("Get Answer") and question:
        with st.spinner("Thinking..."):
            answer = generate_answer(question, st.session_state.chunks, st.session_state.index)
        st.write("### Answer")
        st.write(answer)