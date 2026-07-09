import streamlit as st
import os
import hashlib
import pickle
import numpy as np
import faiss
import fitz  # PyMuPDF
from fastembed import TextEmbedding
from groq import Groq
from dotenv import load_dotenv

# ----------------------------
# Load API Key
# ----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ----------------------------
# Streamlit Config
# ----------------------------
st.set_page_config(
    page_title="PDF Chat AI (RAG) - Fast",
    page_icon="⚡",
    layout="wide"
)

CACHE_DIR = ".rag_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ----------------------------
# Cached Resources
# Loaded ONCE at app start, separate from PDF processing spinner
# ----------------------------
@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def load_embedding_model():
    return TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_groq_client(api_key):
    return Groq(api_key=api_key)

def chunk_text(text, chunk_size=1500, overlap=150):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= length:
            break
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]

def embed_texts(model, texts):
    vectors = np.array(list(model.embed(texts)), dtype="float32")
    faiss.normalize_L2(vectors)
    return vectors

# Load model early, not hidden inside PDF processing step
embed_model = load_embedding_model()

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:

    st.title("⚙️ Configuration")

    groq_api_key_input = st.text_input(
        "Enter Groq API Key",
        type="password"
    )

    if groq_api_key_input:
        GROQ_API_KEY = groq_api_key_input
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY

    if GROQ_API_KEY:
        st.success("Groq API Connected ✅")

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type="pdf"
    )

# ----------------------------
# Main UI
# ----------------------------
st.title("⚡ Fast PDF RAG Assistant")
st.caption("Upload your PDF and ask questions — optimized for speed.")

if not GROQ_API_KEY:
    st.warning("Please enter your Groq API Key.")
    st.stop()

if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

# ----------------------------
# PDF Processing (with disk cache)
# ----------------------------
if uploaded_file:

    pdf_bytes = uploaded_file.read()
    current_hash = hashlib.md5(pdf_bytes).hexdigest()

    if st.session_state.file_hash != current_hash:

        index_path = os.path.join(CACHE_DIR, f"{current_hash}.index")
        chunks_path = os.path.join(CACHE_DIR, f"{current_hash}.pkl")

        if os.path.exists(index_path) and os.path.exists(chunks_path):
            # Already processed before -> instant load, no re-embedding
            index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
        else:
            with st.spinner("Processing PDF..."):
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                raw_text = "".join(page.get_text() for page in doc)
                doc.close()

                chunks = chunk_text(raw_text)
                vectors = embed_texts(embed_model, chunks)

                index = faiss.IndexFlatIP(vectors.shape[1])
                index.add(vectors)

                faiss.write_index(index, index_path)
                with open(chunks_path, "wb") as f:
                    pickle.dump(chunks, f)

        st.session_state.index = index
        st.session_state.chunks = chunks
        st.session_state.file_hash = current_hash

# ----------------------------
# Chat
# ----------------------------
if st.session_state.index is not None:

    user_query = st.text_input(
        "Ask a question about your PDF"
    )

    if user_query:

        q_vector = embed_texts(embed_model, [user_query])

        k = min(3, len(st.session_state.chunks))
        scores, idxs = st.session_state.index.search(q_vector, k)

        context = "\n\n".join(
            st.session_state.chunks[i] for i in idxs[0] if i != -1
        )

        st.markdown("## 📄 Answer")
        answer_box = st.empty()
        full_answer = ""

        try:
            client = load_groq_client(GROQ_API_KEY)

            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": f"""
Answer ONLY using the context below.
If the answer is not available in the context, say:
"I couldn't find this information in the uploaded PDF."

Context:
{context}
"""
                    },
                    {
                        "role": "user",
                        "content": user_query
                    }
                ],
                stream=True
            )

            for part in stream:
                delta = part.choices[0].delta.content or ""
                full_answer += delta
                answer_box.markdown(full_answer + "▌")

            answer_box.markdown(full_answer)

        except Exception as e:
            st.error(f"Error: {e}")