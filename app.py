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
