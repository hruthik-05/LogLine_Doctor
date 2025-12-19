import os
import time
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# Stability AI
import stability_sdk.client as stability
from stability_sdk import interfaces

# AutoGen
from autogen.agentchat import AssistantAgent, UserProxyAgent

# LangChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---------------------- Load Environment Variables ----------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
stability_api_key = os.getenv("STABILITY_API_KEY")

if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env file.")
if not stability_api_key:
    raise ValueError("❌ STABILITY_API_KEY not found in .env file.")

# ---------------------- LLM Configuration ----------------------
config_list = [
    {
        "model": "llama-3.1-8b-instant",
        "api_key": groq_api_key,
        "base_url": "https://api.groq.com/openai/v1",
    }
]
llm_config = {"config_list": config_list}

# ---------------------- Stability AI Client ----------------------
stability_client = stability.StabilityInference(key=stability_api_key, verbose=True)

# ---------------------- 1. Logline Critique Tool ----------------------
def critique_logline(logline: str) -> str:
    """
    Analyzes and critiques a logline based on storytelling principles.
    Uses both a RAG (retrieval-augmented) context from ChromaDB (if available)
    and a static principles text file for structured feedback.
    """

    # --- 1. Load Embeddings + Optional RAG Context ---
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    try:
        retrieved_docs = vector_store.similarity_search(logline, k=4)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    except Exception:
        context = "No additional RAG context found. Focus only on the story structure."

    # --- 2. Load Logline Principles (Grounding File) ---
    try:
        with open("logline_principles.txt", "r") as f:
            principles = f.read()
    except FileNotFoundError:
        principles = """
        A good logline should include:
        1. The PROTAGONIST – Who is the story about?
        2. The GOAL – What do they want or need?
        3. The CONFLICT – What stands in their way?
        4. The STAKES – What happens if they fail?
        5. The TONE – What kind of story is this (drama, thriller, comedy, etc.)?

        Rules:
        - Keep the original idea intact.
        - Don’t invent unrelated people or settings.
        - Keep it under 2 sentences.
        - Make it emotionally engaging.
        """

    # --- 3. LLM Configuration ---
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.6,
        api_key=groq_api_key
    )

    # --- 4. Build Prompt ---
    prompt = PromptTemplate.from_template("""
You are a professional **screenplay analyst**. 
Critique the following logline using these Logline Principles:

{principles}

If RAG context is available, you may use it for reference:
{context}

Now, provide a short but sharp critique (3-5 bullet points) focusing on:
- The protagonist’s clarity and motivation
- The strength of the conflict and stakes
- The originality and emotional pull
- How it could be rewritten to sound more cinematic

User Logline: {logline}
""")

    # --- 5. Run the LLM Chain ---
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run({
        "principles": principles,
        "context": context,
        "logline": logline
    })

    return result.strip()

# ---------------------- 2. Image Generation Tool ----------------------
def generate_image(prompt: str, style_preset: str = "photographic") -> str:
    filename = f"image_output_{int(time.time())}.png"
    full_prompt = (
        f"Film poster concept art, cinematic lighting, high detail, {style_preset}. "
        f"Subject: {prompt}"
    )

    try:
        answers = stability_client.generate(
            prompt=full_prompt,
            steps=30,
            cfg_scale=8.0,
            width=1024,
            height=576,
            samples=1,
            seed=42
        )

        for resp in answers:
            for artifact in getattr(resp, "artifacts", []):
                # Handle any SDK version: check success by type or name
                if hasattr(artifact, "binary") and artifact.binary:
                    img_data = artifact.binary
                    img = Image.open(BytesIO(img_data))
                    img.save(filename)
                    img.show()  # 👈 Automatically open image after saving
                    print(f"✅ Image saved as {filename}")
                    return filename

        return "ERROR: No successful image artifact returned."

    except Exception as e:
        return f"ERROR generating image: {e}"


# ---------------------- 3. Setup AutoGen Agents ----------------------
analyst_agent = AssistantAgent(
    name="AnalystAgent",
    system_message= "You are a film development analyst. Use the critique_logline tool only. "
    "Your feedback must stay focused on the **original premise**, "
    "not suggesting new stories. Highlight missing elements (character, goal, conflict, stakes) clearly.",
    llm_config=llm_config,
)

creative_writer_agent = AssistantAgent(
    name="CreativeWriterAgent",
    system_message="An isolated man discovers a dark connection with a charismatic serial killer, forcing him to confront the blurred lines between love and violence in a twisted game of fate.",
    llm_config=llm_config,
)

image_agent = AssistantAgent(
    name="ImageAgent",
    system_message="You are an AI artist. Use the generate_image tool and return only the saved file path.",
    llm_config=llm_config,
)

user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=6,
    code_execution_config={"use_docker": False},
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "") if isinstance(x, dict) else False,
)

# Register tools with UserProxyAgent
user_proxy.register_for_execution(critique_logline)
user_proxy.register_for_execution(generate_image)

# ---------------------- 4. Workflow Function ----------------------
def run_logline_doctor(bad_logline: str):
    print("🔍 Step 1: Analyzing logline...")
    user_proxy.initiate_chat(analyst_agent, message=f"Analyze: {bad_logline}", max_turns=2)
    critique = user_proxy.last_message(analyst_agent)["content"]

    print("✍️ Step 2: Rewriting logline...")
    rewrite_task = f"Original Logline: {bad_logline}\nCritique: {critique}\nRewrite the logline."
    user_proxy.initiate_chat(creative_writer_agent, message=rewrite_task, max_turns=1)
    final_logline = user_proxy.last_message(creative_writer_agent)["content"].strip().replace('"', '')

    print("🎨 Step 3: Generating image...")
    # Here, generate_image function will actually be executed by the user_proxy
    image_path = generate_image(final_logline)

    return {
        "original_logline": bad_logline,
        "critique": critique,
        "final_logline": final_logline,
        "image_path": image_path
    }

# ---------------------- 5. Run the Logline Doctor ----------------------
if __name__ == "__main__":
    bad_logline = "an old woman sacrificed everything for her family"
    results = run_logline_doctor(bad_logline)

    print("\n--- ✅ FINAL RESULT ---")
    print(f"Original Logline: {results['original_logline']}\n")
    print(f"🧩 Critique: {results['critique']}\n")
    print(f"✍️ Rewritten Logline: {results['final_logline']}\n")
    print(f"🖼️ Image saved at: {os.path.abspath(results['image_path'])}")
