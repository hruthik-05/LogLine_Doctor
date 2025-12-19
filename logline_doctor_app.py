import streamlit as st
import os
import sys
from dotenv import load_dotenv

# --- AutoGen (v0.7.5+) ---
from autogen.agentchat import AssistantAgent, UserProxyAgent

# --- LangChain Integrations ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate 
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- RAG/Ingestion Constants ---
CHROMA_DB_DIR = "./chroma_db"
DATA_FILE = "logline_principles.txt"

# --- 1. Load API Key & LLM Configuration ---
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    # Use st.error for the Streamlit environment
    st.error("GROQ_API_KEY not found in .env file. Please check your .env file.")
    st.stop() # Stop the Streamlit app execution

# LLM Config for AutoGen Agents
config_list = [
    {
        "model": "llama-3.1-8b-instant",
        "api_key": api_key,
        "base_url": "https://api.groq.com/openai/v1",
        "price": [0.0, 0.0]
    }
]
llm_config = {"config_list": config_list}

# =======================================================
# --- BACKEND LOGIC (from app.py and ingest_data.py) ---
# =======================================================

def ingest_data_for_ui():
    """Helper function to run the RAG data ingestion (logic from ingest_data.py)."""
    if not os.path.exists(DATA_FILE):
         raise FileNotFoundError(f"Missing RAG principles file: {DATA_FILE}. Please create it.")
         
    loader = TextLoader(DATA_FILE)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Overwrite or create the database
    Chroma.from_documents(
        texts, 
        embeddings, 
        persist_directory=CHROMA_DB_DIR
    )
    return True

def critique_logline(logline: str) -> str:
    """
    Critique a movie logline using RAG and Groq LLM.
    This function acts as a tool called by the AnalystAgent.
    """
    # Note: In Streamlit, print statements go to the terminal
    print(f"\n[Tool Call: critique_logline] Received logline: {logline}")

    # --- RAG Setup ---
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

    # Retrieve relevant principles
    retrieved_docs = vector_store.similarity_search(logline, k=4)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    print(f"[RAG Function] Retrieved context length: {len(context)} characters")

    # --- Groq LLM + Prompt (modern LangChain style) ---
    langchain_llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.7,
        api_key=api_key
    )
    rag_prompt = PromptTemplate.from_template("""
    Based *only* on the following Logline Principles (RAG Context):
    {context}

    Please critique the user-provided logline below. Your critique MUST focus on the four key principles 
    (Protagonist, Goal, Conflict, Stakes) and state exactly what is weak or missing for each.
    
    User Logline: {logline}
    """)
    
    # Runnable-style chain
    rag_chain = rag_prompt | langchain_llm
    critique = rag_chain.invoke({"context": context, "logline": logline})
    
    return critique.content if hasattr(critique, "content") else str(critique)


# --- 3. Define Autogen Agents (Defined here for the run_logline_doctor function) ---

# Analyst Agent (Tool User)
analyst_agent = AssistantAgent(
    name="AnalystAgent",
    system_message="""You are a script analyst. Your sole purpose is to analyze a movie logline.
You MUST call the `critique_logline` tool with the user's logline as the argument.
The tool's output is your final message; do not add extra text or commentary.
""",
    llm_config=llm_config
)

# Creative Writer Agent (Receiver of Critique)
creative_writer_agent = AssistantAgent(
    name="CreativeWriterAgent",
    system_message="""You are an expert Hollywood scriptwriter.
You will be given an original (bad) logline and a critique of it.
Your job is to rewrite the logline to be powerful, compelling, and commercially viable,
addressing all the points in the critique.
Reply with *only* the new, rewritten logline.
""",
    llm_config=llm_config,
)

# User Proxy Agent (Initiator and Tool Executor)
# We make this a persistent variable or define it outside the function
user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=4,
    code_execution_config={"use_docker": False},
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "") if isinstance(x, dict) else False,
)

# Register the function once
user_proxy.register_for_execution(critique_logline)


def run_logline_doctor(bad_logline: str):
    """Orchestrates the two-step agent workflow (logic from app.py)."""
    
    # Step 1: Analyst Critique (Tool Use)
    user_proxy.initiate_chat(
        analyst_agent,
        message=f"Please analyze this logline: {bad_logline}",
        max_turns=2 
    )

    # Extract the critique from the analyst's final message
    critique = user_proxy.last_message(analyst_agent)["content"]
    
    print(f"\n[Main] Received Critique from Analyst:\n{critique}")

    # Step 2: Creative Rewrite
    rewrite_task = f"""
Original Logline: {bad_logline}

Analyst's Critique (Use this to guide the rewrite):
{critique}

Please rewrite this logline.
"""
    # Initiate chat with the writer agent
    user_proxy.initiate_chat(
        creative_writer_agent,
        message=rewrite_task,
        max_turns=1 
    )

    final_logline = user_proxy.last_message(creative_writer_agent)["content"]
    
    # Return the results as a structured dictionary for the Streamlit app
    return {
        "original_logline": bad_logline,
        "critique": critique,
        "final_logline": final_logline.strip()
    }


# =======================================================
# --- STREAMLIT UI (logline_doctor_app.py) ---
# =======================================================

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="The Logline Doctor 🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Session State Management ---
if 'ingested' not in st.session_state:
    st.session_state['ingested'] = False
if 'results' not in st.session_state:
    st.session_state['results'] = None

# --- UI Components ---
st.title("🎬 The Logline Doctor: AI Agent Team")
st.caption("Powered by **AutoGen**, **Groq (Llama 3.1)**, and **Local RAG (ChromaDB)**")

st.markdown("""
### Fix Your Pitch in Seconds
Enter a weak, one-sentence movie idea. The **Analyst Agent** will critique it against proven screenwriting principles (via RAG), and the **Creative Writer Agent** will rewrite it into a powerful, marketable logline.
""")
st.divider()

# --------------------------
# --- Data Ingestion & Setup ---
# --------------------------
st.sidebar.header("Setup")

chroma_db_exists = os.path.exists(CHROMA_DB_DIR)

if not chroma_db_exists or not st.session_state['ingested']:
    st.sidebar.warning("RAG Database not found! Please ingest data first.")
    if st.sidebar.button("💾 Run Data Ingestion (Setup RAG)"):
        with st.spinner('Ingesting data... This only needs to run once.'):
            try:
                ingest_data_for_ui()
                st.session_state['ingested'] = True
                st.success("Data Ingestion Complete! Ready to analyze.")
                # FIXED: Use st.rerun()
                st.rerun() 
            except FileNotFoundError as fnfe:
                 st.error(f"Ingestion failed: {fnfe}")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
else:
    st.sidebar.success("RAG Database is ready.")
    st.session_state['ingested'] = True

st.sidebar.markdown("""
---
### Instructions
1.  **Set Up:** Ensure your `logline_principles.txt` and `.env` (with `GROQ_API_KEY`) are in the project folder.
2.  **Ingest Data:** Click the **'Run Data Ingestion'** button once to create the local RAG vector store (`chroma_db`).
3.  **Run App:** Enter a logline and click the main button!
""")

st.divider()

# ----------------------
# --- Main Input Form ---
# ----------------------
logline_input = st.text_area(
    "Enter Your Weak Logline:",
    value="A man travels in time to save his family.",
    height=100,
    disabled=not st.session_state['ingested']
)

# Button to trigger the agent workflow
if st.button("🩺 Doctor My Logline!", type="primary", disabled=not st.session_state['ingested']):
    if not logline_input.strip():
        st.error("Please enter a logline to be analyzed.")
    else:
        st.session_state['results'] = None
        
        with st.spinner(f"Agents collaborating on '{logline_input}'... (This involves Groq API calls and RAG)"):
            try:
                results = run_logline_doctor(logline_input)
                st.session_state['results'] = results
                st.session_state['original_logline'] = logline_input
            except Exception as e:
                st.error(f"An error occurred during agent collaboration: {e}")
                st.info("Check your terminal for detailed traceback.")
                st.session_state['results'] = None

st.divider()

# ----------------------
# --- Results Display ---
# ----------------------
if st.session_state['results']:
    results = st.session_state['results']
    original = st.session_state['original_logline']
    
    st.header("✅ Analysis Complete!")
    
    # 1. Final Rewritten Logline
    st.subheader("Final Polished Logline")
    st.code(results['final_logline'], language="markdown")
    st.success("This is the marketable pitch ready for Hollywood.")
    
    st.markdown("---")
    
    # 2. Agent Workflow Details (Critique)
    st.subheader("Agent Workflow Breakdown")
    
    # Original Logline
    st.info(f"**Original Logline:** *{original}*")
    
    # Analyst Critique
    with st.expander("🔬 Analyst Agent's RAG-Based Critique"):
        st.write("**Critique received from the Analyst Agent:**")
        st.markdown(results['critique'])
    
    # Creative Writer Task
    with st.expander("✍️ Creative Writer Agent's Task"):
        st.write("**The Writer Agent used the critique below to generate the new logline.**")
        st.code(results['critique'], language="text")