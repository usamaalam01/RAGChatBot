# RAG Document Q&A Application - Development Guide

This guide explains how to build a Retrieval-Augmented Generation (RAG) application from scratch using Python, LangChain, and Streamlit.

---

## Table of Contents

1. [What is RAG?](#1-what-is-rag)
2. [Application Overview](#2-application-overview)
3. [Prerequisites](#3-prerequisites)
4. [Project Setup](#4-project-setup)
5. [Development Steps](#5-development-steps)
6. [Configuration](#6-configuration)
7. [Running the Application](#7-running-the-application)
8. [Deployment](#8-deployment)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances Large Language Models (LLMs) by providing them with relevant context from external documents before generating responses.

### How RAG Works:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  User Query │────>│   Retriever  │────>│  Retrieved  │
└─────────────┘     │ (Vector DB)  │     │   Context   │
                    └──────────────┘     └──────┬──────┘
                                                │
                                                v
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Answer    │<────│     LLM      │<────│   Prompt    │
└─────────────┘     └──────────────┘     │  (Query +   │
                                         │   Context)  │
                                         └─────────────┘
```

### Key Benefits:
- **Grounded responses**: Answers are based on your documents, not just the LLM's training data
- **Up-to-date information**: Use current documents without retraining the model
- **Reduced hallucination**: LLM has specific context to reference
- **Source attribution**: You can show users where the answer came from

---

## 2. Application Overview

This application allows users to:
- Upload documents (PDF, DOCX, CSV)
- Store document embeddings in a vector database
- Ask natural language questions about the documents
- Receive AI-generated answers with source references

### Technology Stack:

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | Streamlit | Web UI |
| RAG Framework | LangChain | Orchestration |
| Vector Database | ChromaDB | Document storage & retrieval |
| LLM Provider | Groq | Fast inference |
| LLM Model | Llama 3.3 70B | Answer generation |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) | Text to vectors |

### Project Structure:

```
RAG_Sample/
├── app.py              # Main application (all logic)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── .env                # Your configuration (not in git)
├── .gitignore          # Git ignore rules
├── chroma_db/          # Vector database (created at runtime)
└── venv/               # Virtual environment
```

---

## 3. Prerequisites

### Required Software:

1. **Python 3.9+** - Download from [python.org](https://www.python.org/downloads/)
2. **Git** (optional) - For version control

### Required Accounts:

1. **Groq API Key** (Free)
   - Go to [console.groq.com](https://console.groq.com)
   - Sign up for a free account
   - Navigate to API Keys and create a new key
   - Copy and save your API key securely

### Verify Python Installation:

```bash
python --version
# Should output: Python 3.9.x or higher
```

---

## 4. Project Setup

### Step 4.1: Create Project Directory

```bash
# Create and enter project directory
mkdir RAG_Sample
cd RAG_Sample
```

### Step 4.2: Create Virtual Environment

A virtual environment isolates your project dependencies from other Python projects.

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

Your terminal should now show `(venv)` at the beginning of the prompt.

### Step 4.3: Create requirements.txt

Create a file named `requirements.txt` with the following content:

```txt
streamlit
langchain
langchain-community
langchain-text-splitters
langchain-groq
langchain-huggingface
sentence-transformers
langchain-chroma
chromadb
pypdf
python-docx
python-dotenv
```

### Step 4.4: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **streamlit**: Web application framework
- **langchain & related packages**: RAG orchestration framework
- **sentence-transformers**: For generating text embeddings
- **chromadb**: Vector database for storing and searching embeddings
- **pypdf**: PDF file reading
- **python-docx**: Word document reading
- **python-dotenv**: Environment variable management

### Step 4.5: Create Environment Configuration

Create `.env.example` (template for others):

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
HF_EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./chroma_db
```

Create `.env` (your actual configuration):

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_MODEL=llama-3.3-70b-versatile
HF_EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./chroma_db
```

**Important**: Replace `gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx` with your actual Groq API key.

### Step 4.6: Create .gitignore

Create a `.gitignore` file to prevent sensitive files from being committed:

```gitignore
# Virtual environment
venv/

# Environment variables
.env

# ChromaDB persistence
chroma_db/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo
```

---

## 5. Development Steps

Create `app.py` and follow these steps to build the application:

### Step 5.1: Imports and Configuration

```python
"""
RAG Application - Document Q&A with LangChain and Streamlit
A simple, beginner-friendly RAG application.
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Document loaders
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader
```

**Explanation:**
- `streamlit`: Creates the web interface
- `dotenv`: Loads environment variables from `.env` file
- `langchain_groq`: Connects to Groq's LLM API
- `langchain_huggingface`: Provides HuggingFace embeddings
- `langchain_chroma`: ChromaDB vector store integration
- `langchain_text_splitters`: Splits documents into smaller chunks
- `langchain_core`: Core LangChain components (prompts, parsers, chains)
- `langchain_community.document_loaders`: Loaders for various file types

### Step 5.2: Load Configuration

```python
# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 3

# RAG Prompt template
RAG_PROMPT = """Answer the question based only on the following context:

{context}

Question: {question}

Answer: """
```

**Explanation:**
- `load_dotenv()`: Loads variables from `.env` file
- `os.getenv()`: Gets environment variables with optional defaults
- `CHUNK_SIZE`: Maximum characters per document chunk (1000)
- `CHUNK_OVERLAP`: Characters overlap between chunks (200) - ensures context isn't lost at chunk boundaries
- `RETRIEVAL_K`: Number of relevant chunks to retrieve (3)
- `RAG_PROMPT`: Template that instructs the LLM to answer based only on provided context

### Step 5.3: Initialize Core Components

```python
# =============================================================================
# INITIALIZE COMPONENTS
# =============================================================================

@st.cache_resource
def get_embeddings():
    """Initialize HuggingFace embeddings."""
    return HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)

@st.cache_resource
def get_llm():
    """Initialize Groq LLM."""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0
    )

def get_vectorstore():
    """Get or create ChromaDB vector store."""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name="rag_documents",
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    return vectorstore
```

**Explanation:**
- `@st.cache_resource`: Streamlit decorator that caches expensive resources (embeddings model, LLM) across reruns
- `HuggingFaceEmbeddings`: Converts text to numerical vectors (384 dimensions for MiniLM)
- `ChatGroq`: Interface to Groq's fast LLM API
- `temperature=0`: Makes responses deterministic (no randomness)
- `Chroma`: Vector database that stores and searches embeddings
- `persist_directory`: Saves the vector store to disk for persistence

### Step 5.4: Document Processing Functions

```python
# =============================================================================
# DOCUMENT PROCESSING
# =============================================================================

def load_document(file):
    """Load a single document based on file type."""
    # Save uploaded file to temp location
    suffix = os.path.splitext(file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
        elif suffix == ".csv":
            loader = CSVLoader(tmp_path)
        else:
            st.error(f"Unsupported file type: {suffix}")
            return []

        documents = loader.load()

        # Add source filename to metadata
        for doc in documents:
            doc.metadata["source"] = file.name

        return documents
    finally:
        # Clean up temp file
        os.unlink(tmp_path)
```

**Explanation:**
- Takes a Streamlit uploaded file object
- Saves it to a temporary file (required by LangChain loaders)
- Selects appropriate loader based on file extension
- Loads document content into LangChain `Document` objects
- Adds source filename to metadata for reference
- Cleans up temporary file after loading

```python
def process_documents(files):
    """Load and chunk multiple documents."""
    all_documents = []

    for file in files:
        docs = load_document(file)
        all_documents.extend(docs)

    if not all_documents:
        return []

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(all_documents)

    return chunks
```

**Explanation:**
- Processes multiple uploaded files
- `RecursiveCharacterTextSplitter`: Intelligently splits text by trying to break at paragraphs, sentences, then words
- `chunk_size=1000`: Each chunk is approximately 1000 characters
- `chunk_overlap=200`: 200 character overlap ensures context continuity

```python
def add_documents_to_vectorstore(chunks):
    """Add document chunks to the vector store."""
    if not chunks:
        return

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)

def clear_vectorstore():
    """Clear all documents from the vector store."""
    vectorstore = get_vectorstore()
    # Delete the collection and recreate it
    vectorstore.delete_collection()
    # Recreate empty collection
    get_vectorstore()
```

**Explanation:**
- `add_documents`: Embeds chunks and stores them in ChromaDB
- `clear_vectorstore`: Removes all documents (useful for starting fresh)

### Step 5.5: RAG Chain Implementation

```python
# =============================================================================
# RAG CHAIN
# =============================================================================

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_answer(question):
    """Get answer for a question using RAG."""
    vectorstore = get_vectorstore()

    # Check if vector store has documents
    if vectorstore._collection.count() == 0:
        return None, []

    # Get retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    # Retrieve relevant documents
    sources = retriever.invoke(question)

    if not sources:
        return None, []

    # Build the RAG chain
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    chain = (
        {"context": lambda x: format_docs(sources), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Get the answer
    answer = chain.invoke(question)

    return answer, sources
```

**Explanation:**

1. **Retrieval Phase:**
   - `as_retriever()`: Converts vector store to a retriever interface
   - `search_kwargs={"k": 3}`: Returns top 3 most similar chunks
   - `retriever.invoke(question)`: Performs semantic search for relevant chunks

2. **Generation Phase:**
   - `ChatPromptTemplate`: Creates a structured prompt from template
   - The chain uses LangChain Expression Language (LCEL):
     - `{"context": ..., "question": ...}`: Prepares input dictionary
     - `| prompt`: Formats the prompt with context and question
     - `| llm`: Sends to LLM for answer generation
     - `| StrOutputParser()`: Extracts string response

3. **Return Values:**
   - `answer`: The generated response from the LLM
   - `sources`: The retrieved document chunks (for showing sources)

### Step 5.6: Streamlit User Interface

```python
# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    st.set_page_config(page_title="RAG Document Q&A", page_icon="📚")

    st.title("📚 RAG Document Q&A")
    st.markdown("Upload documents and ask questions about them.")

    # Check for API key
    if not GROQ_API_KEY:
        st.error("Please set GROQ_API_KEY in your .env file")
        st.stop()

    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "docx", "csv"],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    chunks = process_documents(uploaded_files)
                    if chunks:
                        add_documents_to_vectorstore(chunks)
                        st.success(f"Added {len(chunks)} chunks from {len(uploaded_files)} file(s)")
                    else:
                        st.warning("No content extracted from files")

        st.divider()

        # Show document count
        vectorstore = get_vectorstore()
        doc_count = vectorstore._collection.count()
        st.metric("Chunks in Database", doc_count)

        # Clear button
        if doc_count > 0:
            if st.button("🗑️ Clear All Documents", type="secondary"):
                clear_vectorstore()
                st.success("All documents cleared")
                st.rerun()

    # Main chat area
    st.header("💬 Ask a Question")

    question = st.text_input("Enter your question:", placeholder="What is this document about?")

    if question:
        vectorstore = get_vectorstore()
        if vectorstore._collection.count() == 0:
            st.warning("Please upload and process documents first.")
        else:
            with st.spinner("Thinking..."):
                answer, sources = get_answer(question)

            if answer:
                st.subheader("Answer")
                st.write(answer)

                # Display sources
                if sources:
                    st.subheader("📄 Sources")
                    for i, doc in enumerate(sources, 1):
                        with st.expander(f"Source {i}: {doc.metadata.get('source', 'Unknown')}"):
                            st.write(doc.page_content)
            else:
                st.error("Could not generate an answer.")

if __name__ == "__main__":
    main()
```

**Explanation:**

1. **Page Configuration:**
   - `st.set_page_config()`: Sets browser tab title and icon
   - API key validation prevents app from running without configuration

2. **Sidebar (Document Management):**
   - `st.file_uploader()`: Allows multiple file uploads
   - Process button triggers document chunking and embedding
   - Shows current chunk count with `st.metric()`
   - Clear button removes all documents

3. **Main Area (Q&A):**
   - Text input for questions
   - Displays answer in main area
   - Shows source documents in expandable sections

---

## 6. Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | - | Your Groq API key |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Groq model to use |
| `HF_EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHROMA_PERSIST_DIR` | No | `./chroma_db` | Vector database location |

### Hardcoded Settings

| Setting | Value | Description |
|---------|-------|-------------|
| `CHUNK_SIZE` | 1000 | Maximum characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `RETRIEVAL_K` | 3 | Number of chunks to retrieve |

### Available Groq Models

You can change `GROQ_MODEL` to any of these:

| Model | Best For |
|-------|----------|
| `llama-3.3-70b-versatile` | General purpose (recommended) |
| `llama-3.1-8b-instant` | Faster responses, simpler queries |
| `mixtral-8x7b-32768` | Longer context window |

---

## 7. Running the Application

### Local Development

```bash
# Ensure virtual environment is activated
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Run the application
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### How to Use:

1. **Upload Documents**: Use the sidebar to upload PDF, DOCX, or CSV files
2. **Process**: Click "Process Documents" to embed and store them
3. **Ask Questions**: Type your question in the main area
4. **View Sources**: Expand source sections to see where the answer came from

---

## 8. Deployment

### Option 1: Streamlit Community Cloud (Recommended for beginners)

**Free hosting for Streamlit apps**

1. Push your code to GitHub (ensure `.env` is in `.gitignore`)

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Connect your GitHub repository

4. Add secrets in Streamlit Cloud:
   - Go to App Settings > Secrets
   - Add your environment variables:
   ```toml
   GROQ_API_KEY = "your_api_key_here"
   GROQ_MODEL = "llama-3.3-70b-versatile"
   HF_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
   ```

5. Deploy!

**Note**: Streamlit Community Cloud has ephemeral storage, so the vector database will reset on each deployment. For persistent storage, consider other options.

### Option 2: Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .
COPY .env.example .

# Expose Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
# Build the image
docker build -t rag-app .

# Run the container
docker run -p 8501:8501 \
  -e GROQ_API_KEY=your_api_key \
  -v $(pwd)/chroma_db:/app/chroma_db \
  rag-app
```

### Option 3: Cloud Platforms

**AWS / GCP / Azure:**
- Use a VM instance (EC2, Compute Engine, Azure VM)
- Install Python and dependencies
- Use a process manager like `supervisord` or `systemd`
- Set up reverse proxy with Nginx

**Railway / Render / Fly.io:**
- Connect GitHub repository
- Set environment variables in dashboard
- Deploy automatically

---

## 9. Troubleshooting

### Common Issues

**1. "No module named 'xxx'"**
```bash
# Ensure virtual environment is activated and reinstall
pip install -r requirements.txt
```

**2. "GROQ_API_KEY not set"**
- Check that `.env` file exists in project root
- Verify API key is correctly formatted
- Restart the application after changes

**3. "ChromaDB errors"**
```bash
# Delete the chroma_db folder and restart
rm -rf chroma_db/
streamlit run app.py
```

**4. "Embedding model download fails"**
- First run downloads the model (~90MB)
- Ensure internet connection
- Model is cached in `~/.cache/huggingface/`

**5. "PDF/DOCX files not loading"**
- Ensure `pypdf` and `python-docx` are installed
- Check file is not corrupted
- Try a different file

### Performance Tips

1. **Large Documents**: Split very large files before uploading
2. **Many Documents**: Process in batches of 5-10 files
3. **Slow Responses**: Try a smaller model like `llama-3.1-8b-instant`
4. **Memory Issues**: Reduce `CHUNK_SIZE` or restart the application

---

## Next Steps

After completing this basic application, you can extend it with:

1. **Chat History**: Add conversation memory using LangChain's memory classes
2. **Multiple Collections**: Organize documents by topic or user
3. **Authentication**: Add user login with Streamlit-Authenticator
4. **Better UI**: Use Streamlit's chat components (`st.chat_message`)
5. **Evaluation**: Add metrics to measure answer quality
6. **Hybrid Search**: Combine semantic search with keyword search
7. **Re-ranking**: Use a cross-encoder to re-rank retrieved documents

---

## Resources

- [LangChain Documentation](https://python.langchain.com/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Groq Documentation](https://console.groq.com/docs)
- [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers)

---

*This guide was created to help developers understand and build RAG applications. Feel free to modify and extend the code for your use cases.*
