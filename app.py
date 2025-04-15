import streamlit as st
from rag import RAGEngine
from file_processing import FileHandler
from embedding import EmbeddingManager

# Initial settings
st.set_page_config(page_title="RAG System", layout="wide")

def initialize_session_state():
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = None
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = None
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []  # To store the processed text once

initialize_session_state()

# Application title
st.title("🗨️ RAG Chat System")
st.markdown("---")

# Sidebar for settings and file upload
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Embedding model selection
    embedding_choice = st.selectbox(
        "Embedding Model",
        ["mpnet", "distilroberta", "bert"],
        index=None,
        placeholder="Select embedding model..."
    )
    
    # LLM model selection
    llm_choice = st.selectbox(
        "LLM Model",
        ["llama3", "gemini", "deepseek"],
        index=None,
        placeholder="Select LLM..."
    )
    
    # Button to initialize models
    if st.button("Initialize Models"):
        try:
            embedding_manager = EmbeddingManager(model_name=embedding_choice)
            rag_engine = RAGEngine(embedding_manager=embedding_manager, model=llm_choice)
            
            st.session_state.embedding_model = embedding_choice
            st.session_state.llm_model = llm_choice
            st.session_state.rag_engine = rag_engine
            
            st.success("Models initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
    
    st.markdown("---")
    st.header("📁 Document Upload")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx"],
        accept_multiple_files=True 
    )

    # Process files only once
    if uploaded_files and not st.session_state.file_uploaded:
        try:
            file_handler = FileHandler()
            rag_engine = st.session_state.rag_engine
            
            if rag_engine:
                total_files = len(uploaded_files)
                progress_bar = st.progress(0)
                
                # Process each uploaded file
                for i, uploaded_file in enumerate(uploaded_files):
                    # Extract text from file
                    text = file_handler.extract_text(uploaded_file)
                    
                    # Create a unique document ID
                    doc_id = hash(uploaded_file.name + str(uploaded_file.size))
                    
                    # Create embedding and store in Milvus
                    embeddings = rag_engine.embedding_manager.get_embeddings([text])[0]
                    rag_engine.milvus_db.insert_document(
                        text=text,
                        embedding=embeddings,
                        doc_id=doc_id
                    )
                    
                    # Save the full document text for rag processing (e.g., in both rag_engine.documents and session state)
                    rag_engine.documents.append(text)
                    st.session_state.processed_files.append(text)
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / total_files)
                
                # Mark that files have been processed so they aren’t re-processed on subsequent re-runs
                st.session_state.file_uploaded = True
                st.success(f"{len(uploaded_files)} documents processed successfully!")
            else:
                st.warning("Please initialize models first!")
                
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
    
    st.markdown("---")
    if st.button("🚪 Exit System"):
        if st.session_state.rag_engine:
            st.session_state.rag_engine.milvus_db.close()
        st.rerun()

# Main chat interface
st.header("💬 Chat Interface")

# Display conversation history
for entry in st.session_state.conversation:
    with st.chat_message(entry["role"]):
        st.write(entry["content"])

# Get user input and process query
prompt = st.chat_input("Enter your question...")

if prompt:
    if not st.session_state.file_uploaded:
        st.error("Please upload a document first!")
    elif not st.session_state.rag_engine:
        st.error("Please initialize models first!")
    else:
        # Append user's query to conversation history
        st.session_state.conversation.append({"role": "user", "content": prompt})
        
        # Process query and generate response
        with st.spinner("Generating response..."):
            try:
                # Use the selected RAG method (simple or graph) to handle the query
                response = st.session_state.rag_engine.select_rag_method(prompt)
                answer = response['result']
                
                # Append assistant's answer to conversation history
                st.session_state.conversation.append({"role": "assistant", "content": answer})
                
                # Refresh the app to update chat interface
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
