import streamlit as st
import google.generativeai as genai
import faiss
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

st.set_page_config(page_title="KIET Chatbot", page_icon="🤖", layout="centered")

# ========== Styling ==========
st.markdown(
    """
    <style>
    body { background-color: #0e1117; color: white; }
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
    .stButton > button:hover { background-color: #ff3333; transform: scale(1.05); }
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

# ========== Load and Chunk Text ==========
@st.cache_data
def load_textbook(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

nltk.download('punkt')

def split_text_by_sentences(text, chunk_size=3):
    sentences = sent_tokenize(text)
    return [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

text = load_textbook("kiet_textbook.txt")
chunks = split_text_by_sentences(text)

# ========== Embedding + FAISS ==========
def embed_text(texts):
    return [genai.embed_content(model="models/embedding-001", content=t, task_type="retrieval_document")["embedding"] for t in texts]

@st.cache_resource
def embed_chunks(chunks):
    embeddings = embed_text(chunks)
    faiss_index = faiss.IndexFlatL2(len(embeddings[0]))
    faiss_index.add(np.array(embeddings).astype("float32"))
    return embeddings, faiss_index

chunk_embeddings, index = embed_chunks(chunks)

# ========== Search ==========
def search_semantic(query, chunks, index, top_k=3):
    query_embedding = genai.embed_content(model="models/embedding-001", content=query, task_type="retrieval_query")["embedding"]
    D, I = index.search(np.array([query_embedding]).astype("float32"), top_k)
    return [chunks[i] for i in I[0]]

# ========== UI ==========
if "last_response" not in st.session_state:
    st.session_state.last_response = ""
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

user_query = st.text_input("Enter your question here...", label_visibility="collapsed")

if st.button("Ask") and user_query:
    with st.spinner("Thinking..."):
        context_chunks = search_semantic(user_query, chunks, index)
        context = "\n\n".join(context_chunks)
        prompt = f"""
You are KIET GPT, a smart assistant created to help users learn about KIET College. Use only the provided context below to answer the question as accurately as possible.

If the answer is not found in the context, reply with "I'm not sure based on the provided information."

Context:
{context}

Question:
{user_query}
        """.strip()

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
