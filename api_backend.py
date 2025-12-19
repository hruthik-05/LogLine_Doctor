import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS # Needed for local testing between front and back end

# --- AutoGen ---
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

# --- Flask App Setup ---
app = Flask(__name__)
# Enable CORS for local development
CORS(app) 

# --- 1. Load API Key & LLM Configuration ---
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    # Use standard Python error for the backend script
    raise ValueError("GROQ_API_KEY not found in .env file")

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

# --- RAG Setup (Must run once before the API is used) ---
def ingest_data():
    """Run data ingestion if chroma_db doesn't exist."""
    if os.path.exists(CHROMA_DB_DIR):
        print("RAG Database already exists. Skipping ingestion.")
        return
        
    print("Ingesting data for RAG setup...")
    if not os.path.exists(DATA_FILE):
         raise FileNotFoundError(f"Missing RAG principles file: {DATA_FILE}. Cannot start API.")
         
    loader = TextLoader(DATA_FILE)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    Chroma.from_documents(
        texts, 
        embeddings, 
        persist_directory=CHROMA_DB_DIR
    )
    print("RAG Database successfully created/updated.")

# Run ingestion when the API starts
ingest_data()

# --- Tool Function ---
def critique_logline(logline: str) -> str:
    # Note: Added print for console logging when tool is executed
    print(f"\n[Tool Execution] Critiquing logline: {logline}") 
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    retrieved_docs = vector_store.similarity_search(logline, k=4)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    print(f"[RAG Function] Retrieved context length: {len(context)} characters")

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
    
    rag_chain = rag_prompt | langchain_llm
    critique = rag_chain.invoke({"context": context, "logline": logline})
    
    return critique.content if hasattr(critique, "content") else str(critique)

# --- Autogen Agents (Defined globally for the API) ---
analyst_agent = AssistantAgent(
    name="AnalystAgent", 
    system_message="""You are a script analyst. Your sole function is to call the `critique_logline` tool with the user's logline. Your final response MUST contain ONLY the raw output of the tool, with no other commentary, headers, or conversational text.""", 
    llm_config=llm_config
)
creative_writer_agent = AssistantAgent(
    name="CreativeWriterAgent", 
    system_message="""You are an expert Hollywood scriptwriter. Rewrite the logline to be powerful, compelling, and commercially viable, addressing all points in the critique. Reply with *only* the new, rewritten logline.""", 
    llm_config=llm_config
)
user_proxy = UserProxyAgent(
    name="UserProxy", 
    human_input_mode="NEVER", 
    max_consecutive_auto_reply=4, 
    code_execution_config={"use_docker": False}, 
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "") if isinstance(x, dict) else False
)
user_proxy.register_for_execution(critique_logline)

# --- Workflow Function ---
def run_logline_doctor(bad_logline: str):
    """Orchestrates the two-step agent workflow."""
    
    # Step 1: Analyst Critique (Tool Use)
    user_proxy.initiate_chat(analyst_agent, message=f"Please analyze this logline: {bad_logline}", max_turns=2)
    critique = user_proxy.last_message(analyst_agent)["content"]
    
    # Step 2: Creative Rewrite
    # Pass the raw critique content to the creative writer
    rewrite_task = f"Original Logline: {bad_logline}\nAnalyst's Critique (Use this to guide the rewrite):\n{critique}\n\nPlease rewrite this logline."
    user_proxy.initiate_chat(creative_writer_agent, message=rewrite_task, max_turns=1)
    final_logline = user_proxy.last_message(creative_writer_agent)["content"]
    
    return {
        "original_logline": bad_logline,
        "critique": critique,
        "final_logline": final_logline.strip()
    }

# =======================================================
# --- FLASK ROUTES ---
# =======================================================

@app.route('/analyze', methods=['POST'])
def analyze_logline():
    data = request.get_json()
    logline = data.get('logline', '').strip()

    if not logline:
        return jsonify({"error": "No logline provided."}), 400

    print(f"API received logline: {logline}")
    
    try:
        # Run the core agent logic
        results = run_logline_doctor(logline)
        return jsonify(results)
    except Exception as e:
        print(f"Error during agent execution: {e}")
        return jsonify({"error": "Internal Agent Error", "details": str(e)}), 500

@app.route('/')
def serve_frontend():
    # Flask will look for index.html in the 'templates' folder
    return render_template('index.html')


if __name__ == '__main__':
    print("Starting Logline Doctor API. Ensure chroma_db is built or data is ingested.")
    app.run(debug=True, port=5000)
