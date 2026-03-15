import streamlit as st
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
import uuid

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
# Initialize Groq
# ----------------------------

GROQ_API_KEY = "GROQ_API_KEY"

client = Groq(api_key=GROQ_API_KEY)

# ----------------------------
# Load Embedding Model (Cached)
# ----------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ----------------------------
# Load Vector Database (Cached)
# ----------------------------

@st.cache_resource
def load_vector_db():
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(
        name="mainframe_programs"
    )
    return collection

collection = load_vector_db()

# ----------------------------
# File Upload
# ----------------------------

uploaded_files = st.file_uploader(
    "Upload Mainframe Files",
    type=["cbl", "jcl", "rexx", "txt", "sql"],
    accept_multiple_files=True
)

# ----------------------------
# Chunk Function
# ----------------------------

def split_into_chunks(text, chunk_size=120):

    lines = text.split("\n")

    chunks = []

    for i in range(0, len(lines), chunk_size):

        chunk = "\n".join(lines[i:i + chunk_size])

        chunks.append(chunk)

    return chunks


# ----------------------------
# Store Chunks
# ----------------------------

def store_chunks(file_name, text):

    chunks = split_into_chunks(text)

    for chunk in chunks:

        embedding = embedding_model.encode(chunk).tolist()

        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(uuid.uuid4())],
            metadatas=[{"source": file_name}]
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
# Ask LLM
# ----------------------------

def ask_llm(question):

    context = retrieve_chunks(question)

    if not context.strip():
        return "No relevant code found in uploaded files."

    prompt = f"""
You are a senior mainframe modernization engineer.

You are expert in:
COBOL
JCL
REXX
DB2

Below are relevant code sections:

{context}

User Question:
{question}

Instructions:

1. Explain the logic clearly
2. Identify business rules
3. Identify DB2 tables if present
4. Suggest modernization opportunities
5. Convert logic to Python when applicable
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

            file_text = file.read().decode("utf-8")

            store_chunks(file.name, file_text)

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
    "Ask about COBOL logic, DB2 tables, modernization or generate code"
)

if user_input:

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):

        st.write(user_input)

    with st.spinner("Analyzing mainframe system..."):

        answer = ask_llm(user_input)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):

        st.write(answer)