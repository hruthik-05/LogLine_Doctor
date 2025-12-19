import os
from dotenv import load_dotenv

# --- AutoGen (v0.7.5+) ---
# Note: Autogen Core imports are no longer needed for the ModelInfo object in this setup.
from autogen.agentchat import AssistantAgent, UserProxyAgent

# --- LangChain Integrations ---
# Correct imports for modern LangChain (core, community, partner)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap



# --- 1. Load API Key & LLM Configuration ---
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# LLM Config for AutoGen Agents
# Note: For Groq/OpenAI-compatible endpoints, Autogen needs the config list structure.
config_list = [
    {
        "model": "llama-3.1-8b-instant",
        "api_key": api_key,
        "base_url": "https://api.groq.com/openai/v1",
        "price": [0.0, 0.0]  # prompt and completion cost per 1K tokens (set to 0.0 to disable tracking)
    }
]
llm_config = {"config_list": config_list}


# --- 2. Setup RAG Tool Function ---

def critique_logline(logline: str) -> str:
    """
    Critique a movie logline using RAG and Groq LLM.
    This function acts as a tool called by the AnalystAgent.
    """
    print(f"\n[Tool Call: critique_logline] Received logline: {logline}")

    # --- RAG Setup (Must match ingestdata.py) ---
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

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

    # ✅ Runnable-style chain (no LLMChain needed)
    rag_chain = rag_prompt | langchain_llm

    critique = rag_chain.invoke({"context": context, "logline": logline})
    return critique.content if hasattr(critique, "content") else str(critique)

# --- 3. Define Autogen Agents ---

# Analyst Agent (Tool User)
analyst_agent = AssistantAgent(
    name="AnalystAgent",
    system_message="""You are a script analyst. Your sole purpose is to analyze a movie logline.
You MUST call the `critique_logline` tool with the user's logline as the argument.
The tool's output is your final message; do not add extra text or commentary.
""",
    llm_config=llm_config  # ✅ remove "functions"
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
user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=4,
    code_execution_config={"use_docker": False},
    # ✅ Only terminate when a message explicitly says "TERMINATE"
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "") if isinstance(x, dict) else False,
)

# FIXED: Register the function with the UserProxy so it can execute the tool call.
user_proxy.register_for_execution(critique_logline)

# --- 4. Define the Workflow Function ---

def run_logline_doctor(bad_logline: str):
    """Orchestrates the two-step agent workflow."""
    
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


# --- 5. Start the Chat Flow (CLI Test) ---

if __name__ == "__main__":
    print("Starting the Logline Doctor agent workflow (using Groq)...")
    
    bad_logline = "a guy struck in a tunnel since 24 hrs"
    print(f"User's bad logline: '{bad_logline}'\n")

    results = run_logline_doctor(bad_logline)

    print("\n--- FINAL RESULT ---")
    print(f"Original Logline: {results['original_logline']}")
    print(f"Rewritten Logline: {results['final_logline']}")