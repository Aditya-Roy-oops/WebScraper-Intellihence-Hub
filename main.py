import streamlit as st
import os
import bs4
import shutil
import hashlib
import time
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# --- Telemetry & Logging Fix ---
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("posthog").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai").setLevel(logging.CRITICAL)

# --- Page Configuration ---
st.set_page_config(
    page_title="WebScraper | Intelligence Hub", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# LangChain & AI Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Load environment variables
load_dotenv()

# Securely handle API Key
try:
    google_api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
except (FileNotFoundError, ValueError, Exception):
    google_api_key = os.getenv("GOOGLE_API_KEY", "")

os.environ["GOOGLE_API_KEY"] = google_api_key

if google_api_key:
    genai.configure(api_key=google_api_key)

# --- Helper: Robust Embedding Wrapper ---
class FallbackEmbeddings(Embeddings):
    """
    Wraps GoogleEmbeddings to return zero-vectors if the API fails (429/Quota).
    This ensures TEXT is always stored in the DB for keyword search, even if semantic search fails.
    """
    def __init__(self, google_embeddings):
        self.google_embeddings = google_embeddings

    def embed_documents(self, texts):
        try:
            return self.google_embeddings.embed_documents(texts)
        except Exception:
            # Return list of zero vectors (dim 768) so Chroma accepts the text
            return [[0.0] * 768 for _ in texts]

    def embed_query(self, text):
        try:
            return self.google_embeddings.embed_query(text)
        except Exception:
            return [0.0] * 768

# --- Helper: Get Available Models ---
@st.cache_data
def get_available_models():
    """Dynamically fetch models available to the API key."""
    try:
        if not google_api_key:
            return ["gemini-1.5-flash"]
        model_list = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                name = m.name.replace("models/", "")
                if "2.0" in name and "lite" in name: continue
                model_list.append(name)
        def model_priority(name):
            if "2.5-flash-lite" in name: return 0
            if "2.5-flash" in name: return 1
            if "gemini-pro" in name: return 2
            return 3
        sorted_list = sorted(model_list, key=model_priority)
        return sorted_list if sorted_list else ["gemini-2.5-flash-lite"]
    except Exception:
        return ["gemini-2.5-flash-lite", "gemini-pro"]

# --- Helper: Fallback Logic ---
def get_semantic_fallback(vector_store, query):
    """
    Returns (markdown_text, match_found_boolean).
    If Embedding API fails, falls back to Keyword Search.
    """
    docs = []
    method = "Semantic Search"
    
    try:
        # Attempt 1: Semantic Search
        docs = vector_store.similarity_search(query, k=5)
        # If semantic search returns docs with 0 relevance (due to zero-embeddings), check content
        if docs and all(len(d.page_content) < 5 for d in docs): 
            docs = [] # Treat as failure
    except Exception:
        method = "Keyword Match"
        pass
    
    # Attempt 2: Keyword Search (Local Fallback)
    if not docs:
        method = "Keyword Match"
        try:
            collection_data = vector_store.get()
            all_texts = collection_data.get('documents', [])
            all_metas = collection_data.get('metadatas', [])
            
            query_words = query.lower().split()
            scored_docs = []
            
            for text, meta in zip(all_texts, all_metas):
                score = 0
                lower_text = text.lower()
                for word in query_words:
                    if word in lower_text:
                        score += 1
                if score > 0:
                    scored_docs.append((score, Document(page_content=text, metadata=meta)))
            
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            docs = [doc for score, doc in scored_docs[:5]]
        except Exception as e:
            return f"Error in local search: {str(e)}", False

    if not docs:
        return "### ❌ No Specific Matches Found\nI checked the website content using both AI and Keyword search, but couldn't find information specifically matching your query.", False
        
    fallback_text = f"### ⚠️ AI Service Unavailable - Using {method}\n"
    fallback_text += "The generative model is currently busy/slow. Here is the **exact extracted content** from the website that matches your query:\n\n"
    fallback_text += "---\n"
    
    for i, doc in enumerate(docs, 1):
        content = doc.page_content.strip().replace("\n", " ")
        source = doc.metadata.get("source", "Unknown Source")
        highlighted_content = content[:600]
        fallback_text += f"**📄 Result #{i}** (Source: {source})\n"
        fallback_text += f"> {highlighted_content}...\n\n"
        
    fallback_text += "---\n*(Raw data retrieved directly from the source index)*"
    return fallback_text, True

# --- Professional Clean CSS ---
st.markdown("""
    <style>
    .status-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        background-color: #e3f2fd;
        color: #0d47a1;
        border: 1px solid #bbdefb;
        display: inline-block;
        margin-bottom: 15px;
    }
    @media (prefers-color-scheme: dark) {
        .status-badge {
            background-color: #1e3a8a;
            color: #bfdbfe;
            border: 1px solid #172554;
        }
    }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: 600; }
    .sidebar-header { font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem; opacity: 0.9; }
    </style>
""", unsafe_allow_html=True)

# --- State Management ---
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "current_url" not in st.session_state: st.session_state.current_url = ""

# --- Business Logic: The "Clean" Processor ---
def get_vectorstore_from_url(url):
    """Crawls URL and persists embeddings."""
    persist_dir = f"./db/v2_{hashlib.md5(url.encode()).hexdigest()}"
    # Use Robust Wrapper
    base_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = FallbackEmbeddings(base_embeddings)
    
    # 1. Check for valid existing index
    if os.path.exists(persist_dir):
        try:
            vs = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            # Validation: Check if it actually has data
            if len(vs.get()['documents']) > 0:
                return vs
            else:
                # If existing dir is empty (failed prev run), clean it
                shutil.rmtree(persist_dir)
        except Exception:
            shutil.rmtree(persist_dir)

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
        
        loader = WebBaseLoader(web_paths=(url,), header_template=headers)
        data = loader.load()

        if not data or not data[0].page_content.strip():
            return "Critical Error: Could not extract meaningful content."

        # Verification step
        content_length = len(data[0].page_content)
        if content_length < 50:
             return f"Error: Extracted content was too short ({content_length} chars). Site might be blocked."

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        document_chunks = text_splitter.split_documents(data)

        st.write(f"Processing {len(document_chunks)} chunks securely...")
        vector_store = Chroma(embedding_function=embeddings, persist_directory=persist_dir)
        
        batch_size = 5
        total_chunks = len(document_chunks)
        progress_bar = st.progress(0, text="Embedding knowledge base...")
        
        for i in range(0, total_chunks, batch_size):
            batch = document_chunks[i : i + batch_size]
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    vector_store.add_documents(batch)
                    break 
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        if attempt < max_retries - 1:
                            time.sleep(5) 
                            continue
                    # Even if 429 fails retries, FallbackEmbeddings ensures we don't crash here
                    # But if real crash happens, we catch it outside
                    raise e
            progress = min((i + batch_size) / total_chunks, 1.0)
            progress_bar.progress(progress, text=f"Embedding: {int(progress*100)}%")
            time.sleep(1.5) 
            
        progress_bar.empty()
        return vector_store
    except Exception as e:
        return f"System Failure: {str(e)}"

def get_pro_rag_chain(vector_store, model_name="gemini-1.5-flash"):
    """Advanced RAG chain."""
    llm = ChatGoogleGenerativeAI(
        model=model_name, temperature=0.1, convert_system_message_to_human=True,
        max_retries=0, 
        request_timeout=5.0 
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and a user question, rephrase it as a standalone question."),
        MessagesPlaceholder("chat_history"), ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, rephrase_prompt)
    system_instruction = (
        "You are the Official Website Knowledge Assistant for the provided website content.\n"
        "Answer the user's question using EXCLUSIVELY the provided context.\n"
        "If answer is missing, say: 'The answer is not available on the provided website.'\n"
        "CONTEXT:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction), MessagesPlaceholder("chat_history"), ("human", "{input}")
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, qa_chain)

# --- Main UI Layout ---
with st.sidebar:
    st.image("LinkedinBanner.png", use_column_width=True)
    st.markdown("<div class='sidebar-header'>System Control</div>", unsafe_allow_html=True)
    with st.container(border=True):
        url_input = st.text_input("Source Website URL", placeholder="https://aditya-roy-oops.github.io/")
        available_models = get_available_models()
        model_choice = st.selectbox("AI Model", available_models, index=0)

        if st.button("⚡ Index Content", type="primary"):
            if url_input:
                with st.status("Processing Website...", expanded=True) as status:
                    st.write("🔍 Connecting to source...")
                    result = get_vectorstore_from_url(url_input)
                    if isinstance(result, str):
                        status.update(label="Index Failed", state="error")
                        st.error(result)
                    else:
                        st.session_state.vector_store = result
                        st.session_state.current_url = url_input
                        st.session_state.chat_history = []
                        status.update(label="Ready!", state="complete")
                        st.success("Indexing Complete.")
            else:
                st.error("Invalid URL.")

    # --- NEW: Raw Data Inspection ---
    if st.session_state.vector_store:
        st.markdown("---")
        st.markdown("### 🧐 Data Inspector")
        with st.expander("View Extracted Text"):
            try:
                # Retrieve all documents to show what was indexed
                all_data = st.session_state.vector_store.get()
                if all_data and all_data['documents']:
                    full_dump = "\n\n------------------------\n\n".join(all_data['documents'])
                    st.text_area("Full Website Content:", value=full_dump, height=300)
                    st.caption(f"Total Chunks: {len(all_data['documents'])}")
                else:
                    st.warning("Index is empty.")
            except Exception as e:
                st.error(f"Read error: {e}")

    st.markdown("---")
    if st.button("🗑️ Reset Application", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.session_state.current_url = ""
        st.rerun()

st.title("🤖 IntelliChat Engine")
st.markdown("Transforming website content into actionable intelligence through grounded RAG.")

if st.session_state.vector_store is None:
    st.divider()
    cols = st.columns(3)
    with cols[0]: st.markdown("### 🔗 Connect\nInput any website URL."); 
    with cols[1]: st.markdown("### 🧠 Learn\nWe chunk text for understanding."); 
    with cols[2]: st.markdown("### 💬 Chat\nAsk grounded questions."); 
    st.info("👈 **Awaiting source URL to initialize intelligence.**")
else:
    st.markdown(f"<div class='status-badge'>● Grounded: {st.session_state.current_url}</div>", unsafe_allow_html=True)
    
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role): st.markdown(message.content)

    if user_input := st.chat_input("Analyze website data..."):
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with st.chat_message("user"): st.markdown(user_input)

        with st.chat_message("assistant"):
            response_container = st.empty()
            with response_container.container():
                with st.spinner("Analyzing..."):
                    try:
                        # Attempt 1: LLM
                        chain = get_pro_rag_chain(st.session_state.vector_store, model_choice)
                        res = chain.invoke({"chat_history": st.session_state.chat_history, "input": user_input})
                        answer = res["answer"]
                        st.markdown(answer)
                        st.session_state.chat_history.append(AIMessage(content=answer))
                    
                    except Exception as e:
                        # Fallback Logic
                        fallback_msg, found_match = get_semantic_fallback(st.session_state.vector_store, user_input)
                        st.markdown(fallback_msg)
                        st.session_state.chat_history.append(AIMessage(content=fallback_msg))
                        
                        # Extra Button to See All Data if needed
                        if not found_match:
                            st.warning("Since no match was found, you can verify the raw data in the Sidebar 'Data Inspector'.")

st.markdown("<div style='position: fixed; bottom: 10px; right: 20px; opacity: 0.5; font-size: 0.7rem;'>WebScraper | RAG Intelligence </div>", unsafe_allow_html=True)