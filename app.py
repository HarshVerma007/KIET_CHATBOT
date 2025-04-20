import streamlit as st
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np

# ========== CONFIG ==========
st.set_page_config(page_title="KIET Chatbot", page_icon="🤖", layout="centered")

# --- Custom CSS ---
st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }

    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        border: none;
        font-size: 16px;
    }

    .stButton > button:hover {
        background-color: #ff3333;
        color: white;
        transform: scale(1.05);
    }

    .stButton > button:active {
        background-color: #cc2929;
        color: white;
    }

    .stTextInput input {
        border: 1px solid #888;
        border-radius: 8px;
        padding: 0.5em;
        font-size: 16px;
    }

    .response {
        background-color: #262730;
        padding: 1em;
        border-radius: 10px;
        margin-top: 10px;
        font-size: 16px;
    }

    .header-title {
        font-size: 40px;
        font-weight: bold;
        color: white;
        text-align: center;
    }

    .subheader {
        font-size: 18px;
        color: #bbbbbb;
        text-align: center;
        margin-bottom: 2em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ========== Header ==========
st.markdown('<div class="header-title">🤖 KIET GPT</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Ask me anything about KIET!</div>', unsafe_allow_html=True)

# ========== GENAI SETUP ==========
API_KEY = "AIzaSyDer9CWKorC7XpCnZ1XDKPuVRuT8NEus48"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ========== LOAD TEXT ==========
@st.cache_data
def load_textbook(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

text = load_textbook("kiet_textbook.txt")

# ========== SPLIT TEXT ==========
def split_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

chunks = split_text(text)

# ========== VECTORIZE ==========
@st.cache_resource
def vectorize_chunks(chunks):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(chunks).toarray().astype("float32")
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    return vectorizer, index

vectorizer, index = vectorize_chunks(chunks)

# ========== SEARCH ==========
def search(query, vectorizer, index, chunks, top_k=3):
    q_vec = vectorizer.transform([query]).toarray().astype("float32")
    _, I = index.search(q_vec, top_k)
    return [chunks[i] for i in I[0]]

# ========== UI Logic ==========
if "last_response" not in st.session_state:
    st.session_state.last_response = ""
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

user_query = st.text_input("Enter your question here...", label_visibility="collapsed")

if st.button("Ask") and user_query:
    with st.spinner("Thinking..."):
        context_chunks = search(user_query, vectorizer, index, chunks)
        prompt = f"Use the following text to answer the question:\n\n{''.join(context_chunks)}\n\nQuestion: {user_query}"

        try:
            response = model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            answer = f"❌ Error: {str(e)}"

        st.session_state.last_response = answer
        st.session_state.last_query = user_query

# ========== Show Response ==========
if st.session_state.last_response:
    st.markdown(f"**🧑 You:** {st.session_state.last_query}")
    st.markdown(f"<div class='response'><strong>🤖 Bot:</strong><br>{st.session_state.last_response}</div>", unsafe_allow_html=True)
