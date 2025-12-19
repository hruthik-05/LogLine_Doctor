import streamlit as st
import os
import time
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# -------------------- Imports --------------------
import stability_sdk.client as stability
from autogen.agentchat import AssistantAgent, UserProxyAgent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# -------------------- Load Environment --------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
stability_api_key = os.getenv("STABILITY_API_KEY")

if not groq_api_key or not stability_api_key:
    st.error("❌ Please add GROQ_API_KEY and STABILITY_API_KEY in your .env file.")
    st.stop()

# -------------------- Setup Clients --------------------
stability_client = stability.StabilityInference(key=stability_api_key, verbose=True)
config_list = [{"model": "llama-3.1-8b-instant", "api_key": groq_api_key, "base_url": "https://api.groq.com/openai/v1"}]
llm_config = {"config_list": config_list}

# -------------------- Load Visual Vector DB --------------------
def load_or_create_vectorstore(file_path: str, persist_dir: str):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={"device": "cpu"})
    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # Populate DB if empty
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(vector_store._collection.get()) == 0:
            for line in lines:
                vector_store.add_documents([Document(page_content=line)])
    return vector_store

visual_vector = load_or_create_vectorstore("visual_descriptions.txt", "./visual_db")

# -------------------- Logline Critique --------------------
def critique_logline(logline: str) -> str:
    try:
        with open("logline_principles.txt", "r") as f:
            principles = f.read()
    except FileNotFoundError:
        principles = "A logline should have a protagonist, goal, conflict, and stakes."

    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.6, api_key=groq_api_key)
    prompt = PromptTemplate.from_template("""
You are a professional screenplay analyst.
Critique this logline using these principles:

{principles}

Focus on:
- Protagonist clarity
- Goal and motivation
- Conflict and stakes
- Cinematic tone

User Logline: {logline}
""")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"principles": principles, "logline": logline}).strip()

# -------------------- Extract Visual Prompt --------------------
def extract_visual_prompt(logline: str) -> str:
    # Retrieve relevant visuals from vector DB
    retrieved_docs = visual_vector.similarity_search(logline, k=6)
    visual_context = "\n".join([doc.page_content for doc in retrieved_docs])

    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.4, api_key=groq_api_key)
    system_message = """
You are a cinematic visual prompt generator for story-based image creation.

Turn the logline and visual context into a short, vivid, realistic visual description suitable for AI image generation.
Include: protagonist, main setting, mood, key action or conflict, lighting, and atmosphere.
Keep it concise (max 35 words). Avoid abstract or symbolic language.
"""

    prompt = PromptTemplate.from_template(f"""
{system_message}

Visual Context:
{visual_context}

Logline:
{logline}

Output:
""")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"logline": logline}).strip()

# -------------------- Image Generator --------------------
def generate_image(logline: str, style_preset: str = "photographic") -> str:
    filename = f"image_output_{int(time.time())}.png"
    short_prompt = extract_visual_prompt(logline)
    full_prompt = f"Film poster, cinematic lighting, ultra detailed, {style_preset}. Subject: {short_prompt}"

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
                if hasattr(artifact, "binary") and artifact.binary:
                    img = Image.open(BytesIO(artifact.binary))
                    img.save(filename)
                    return filename
        return "ERROR: No valid image returned."
    except Exception as e:
        return f"ERROR generating image: {e}"

# -------------------- Agents --------------------
analyst_agent = AssistantAgent(
    name="AnalystAgent",
    system_message="You are a film analyst. Use the critique_logline tool only. Focus on protagonist, goal, conflict, and stakes.",
    llm_config=llm_config,
)
creative_writer_agent = AssistantAgent(
    name="CreativeWriterAgent",
    system_message="You are a creative logline rewriter. Strengthen clarity, tension, and emotion.",
    llm_config=llm_config,
)
user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=6,
    code_execution_config={"use_docker": False},
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "") if isinstance(x, dict) else False,
)
user_proxy.register_for_execution(critique_logline)
user_proxy.register_for_execution(generate_image)

# -------------------- Logline Doctor Flow --------------------
def run_logline_doctor(bad_logline: str):
    user_proxy.initiate_chat(analyst_agent, message=f"Analyze: {bad_logline}", max_turns=2)
    critique = user_proxy.last_message(analyst_agent)["content"]

    rewrite_task = f"Original Logline: {bad_logline}\nCritique: {critique}\nRewrite it clearly and cinematically."
    user_proxy.initiate_chat(creative_writer_agent, message=rewrite_task, max_turns=1)
    final_logline = user_proxy.last_message(creative_writer_agent)["content"].strip().replace('"', '')

    image_path = generate_image(final_logline)
    return bad_logline, critique, final_logline, image_path

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="🎬 Logline Doctor AI", layout="wide")
st.title("🎬 Logline Doctor AI")
st.write("Refine your film loglines using AI — get critiques, rewrites, and cinematic poster images.")

user_logline = st.text_area("Enter your logline:", placeholder="e.g. An old woman sacrifices everything for her family.")

if st.button("Analyze & Generate"):
    if not user_logline.strip():
        st.warning("Please enter a logline.")
    else:
        with st.spinner("Analyzing your logline... 🎭"):
            original, critique, rewritten, image_path = run_logline_doctor(user_logline)

        st.subheader("🧩 Critique")
        st.write(critique)
        st.subheader("✍️ Rewritten Logline")
        st.success(rewritten)

        if os.path.exists(image_path):
            st.subheader("🎨 Generated Poster Image")
            st.image(image_path, caption="AI Generated Film Poster", use_container_width=True)
        else:
            st.error("⚠️ Image generation failed.")

st.markdown("---")
st.caption("Developed by Hruthik • Powered by LangChain, Groq, and Stability AI 🎥")
