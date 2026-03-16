import streamlit as st
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import os
import io

# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(
    page_title="AI Mainframe Modernization Assistant",
    page_icon="💻",
    layout="wide"
)

st.title("💻 AI Mainframe Modernization Assistant")

st.write(
"""
Analyze legacy **COBOL, JCL, REXX and DB2 systems using AI.
Upload programs, explore dependencies, detect DB2 tables and modernize legacy logic.
"""
)

# ----------------------------
# Groq API
# ----------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# ----------------------------
# Embedding Model
# ----------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ----------------------------
# Vector DB
# ----------------------------

@st.cache_resource
def load_vector_db():
    chroma_client = chromadb.Client()
    return chroma_client.get_or_create_collection(
        name="mainframe_programs"
    )

collection = load_vector_db()

# ----------------------------
# Upload Files
# ----------------------------

uploaded_files = st.file_uploader(
    "Upload Mainframe Files",
    type=["cbl", "jcl", "rexx", "txt", "sql"],
    accept_multiple_files=True
)

# ----------------------------
# Chunk Split
# ----------------------------

def split_into_chunks(text, chunk_size=120):

    lines = text.split("\n")

    chunks = []

    for i in range(0, len(lines), chunk_size):
        chunk = "\n".join(lines[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


# ----------------------------
# Store Chunks
# ----------------------------

def store_chunks(file_name, text):

    chunks = split_into_chunks(text)

    embeddings = embedding_model.encode(chunks).tolist()

    ids = [str(uuid.uuid4()) for _ in chunks]

    metadata = [{"source": file_name} for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadata
    )


# ----------------------------
# Retrieve Context
# ----------------------------

def retrieve_chunks(question):

    query_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    docs = results.get("documents", [])

    if not docs or not docs[0]:
        return ""

    return "\n".join(docs[0])


# ----------------------------
# LLM
# ----------------------------

def ask_llm(question):

    context = retrieve_chunks(question)

    if context.strip():

        prompt = f"""
You are a senior mainframe modernization engineer.
Context:
{context}
User Question:
{question}
Instructions:
- Explain the code
- Identify business rules
- Detect DB2 tables
- Suggest modernization
"""

    else:

        prompt = f"""
You are a mainframe systems expert.
Explain clearly.
Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# ----------------------------
# Process Upload
# ----------------------------

if uploaded_files:
    with st.spinner("Indexing uploaded files..."):
        for file in uploaded_files:
            try:
                # Reset buffer just in case
                file.seek(0)
                bytes_data = file.read()
                text = bytes_data.decode("utf-8", errors="ignore")
                
                if not text.strip():
                    st.warning(f"⚠️ {file.name} appears empty, skipping.")
                    continue
                    
                store_chunks(file.name, text)
                
            except Exception as e:
                st.error(f"File error ({file.name}): {e}")
    
    st.success("Files indexed successfully!")

# ----------------------------
# Chat History
# ----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ----------------------------
# Chat Input
# ----------------------------

user_input = st.chat_input(
    "Ask about COBOL logic, DB2 tables, modernization or general mainframe questions"
)

if user_input:

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Analyzing..."):

        answer = ask_llm(user_input)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.write(answer)
