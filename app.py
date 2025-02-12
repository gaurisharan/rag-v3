import os
import tempfile
import streamlit as st
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.embeddings import HuggingFaceEmbedding
from llama_index.core.llms import Mistral
from llama_index.core.readers import LlamaParseReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine

# Initialize Pinecone connection
pc = Pinecone(api_key=st.secrets.PINECONE_API_KEY)
INDEX_NAME = "ragreader"

# Initialize Hugging Face embeddings (1024 dimensions)
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu"
)

# Initialize Mistral LLM
llm = Mistral(api_key=st.secrets.MISTRAL_API_KEY, model="mistral-tiny")

# Document processing pipeline using LlamaParse
def process_documents(uploaded_files):
    temp_dir = "temp_docs"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save uploaded files temporarily
    file_paths = []
    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        file_paths.append(file_path)
    
    # Use LlamaParse to read and process documents
    reader = LlamaParseReader(api_key=st.secrets.LLAMAPARSE_API_KEY)
    documents = reader.load_data(file_paths)
    
    # Create service context
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    )
    
    # Create and store index
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context
    )
    
    return index

# Initialize query engine
def init_query_engine(index):
    retriever = index.as_retriever(similarity_top_k=3)
    return RetrieverQueryEngine.from_args(retriever, service_context=index.service_context)

# Streamlit UI setup
st.set_page_config(page_title="RAG Chat with LlamaParse", layout="wide")

# Sidebar for document upload and processing
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, etc.)",
        type=["pdf", "docx", "pptx", "xlsx"],
        accept_multiple_files=True
    )
    
    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Processing documents..."):
            index = process_documents(uploaded_files)
            st.session_state.query_engine = init_query_engine(index)
            st.success("Documents processed and indexed!")

# Main chat interface
st.title("📚 Document Chat Assistant")
st.caption(f"Connected to Pinecone index: {INDEX_NAME}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input for chat queries
if prompt := st.chat_input("Ask about your documents"):
    if "query_engine" not in st.session_state:
        st.error("Please upload and process documents first")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                response = st.session_state.query_engine.query(prompt)
                
                # Display answer and sources
                st.markdown(f"**Answer**: {response.response}")
                st.markdown("**Relevant Sources**:")
                for node in response.source_nodes:
                    source = node.metadata.get('file_name', 'Unknown')
                    page = node.metadata.get('page_label', 'N/A')
                    st.markdown(f"- `{source}` (Page {page})")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.response
        })
