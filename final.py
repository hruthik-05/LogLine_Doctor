import streamlit as st
import os
import time
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# -------------------- Imports --------------------
import stability_sdk.client as stability
from stability_sdk import interfaces

from autogen.agentchat import AssistantAgent, UserProxyAgent
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

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

# -------------------- Vector Store Setup --------------------
def load_or_create_vectorstore(file_path, db_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={"device": "cpu"})
    if not os.path.exists(db_path):
        os.makedirs(db_path, exist_ok=True)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = f.read().splitlines()
            data = [line for line in data if line.strip()]
            Chroma.from_texts(data, embeddings, persist_directory=db_path)
    return Chroma(persist_directory=db_path, embedding_function=embeddings)

# -------------------- Load both RAG sources --------------------
logline_vector = load_or_create_vectorstore("logline_principles.txt", "./chroma_db")
visual_vector = load_or_create_vectorstore("visual_descriptions.txt", "./visual_db")

# -------------------- Logline Critique --------------------
def critique_logline(logline: str) -> str:
    try:
        retrieved_docs = logline_vector.similarity_search(logline, k=4)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    except Exception:
        context = "No context found."

    with open("logline_principles.txt", "r") as f:
        principles = f.read()

    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.6, api_key=groq_api_key)
    prompt = PromptTemplate.from_template("""
You are a professional screenplay analyst and story structure coach.
Critique this logline using cinematic storytelling theory.

Principles:
{principles}

Relevant Context:
{context}

Focus on protagonist, goal, conflict, and stakes.
Logline: {logline}
""")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"principles": principles, "context": context, "logline": logline}).strip()

# -------------------- Visual Context Retriever --------------------
def enrich_with_visual_context(logline: str) -> str:
    keywords = [w for w in logline.split() if w.isalpha()]
    visual_contexts = []
    for kw in keywords:
        docs = visual_vector.similarity_search(kw, k=1)
        if docs:
            visual_contexts.append(docs[0].page_content)
    return " ".join(visual_contexts)

# -------------------- Extract Visual Prompt --------------------
def extract_visual_prompt(logline: str, groq_api_key: str) -> str:
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.4, api_key=groq_api_key)
    prompt = PromptTemplate.from_template("""
You are a cinematic visual prompt engineer.
Your task: convert the rewritten logline into a short (≤25 words) movie poster prompt.

Rules:
1. Stay true to the story.
2. Include protagonist, setting, lighting, and action.
3. Avoid adding new details.
4. Add cinematic words like "fog", "glow", "dramatic lighting".

Logline:
{logline}

Example:
"A detective under neon rain, Tokyo skyline glowing, noir lighting, tense atmosphere."
""")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"logline": logline}).strip()

# -------------------- Image Generator --------------------
def generate_image(logline: str, style_preset: str = "photographic") -> str:
    filename = f"image_output_{int(time.time())}.png"

    short_prompt = extract_visual_prompt(logline, groq_api_key)
    visual_context = enrich_with_visual_context(logline)

    full_prompt = (
        f"Cinematic movie poster concept. {short_prompt}. "
        f"Additional visual grounding: {visual_context}. "
        f"Ultra-detailed, realistic, filmic lighting, atmospheric tone, dramatic composition, {style_preset} style."
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
    system_message="""
You are a senior **screenplay analyst and story structure expert** with the precision of Robert McKee and the clarity of Blake Snyder.
You critique film loglines using both classical Hollywood structure and modern storytelling principles.

🧠 Your Core Role:
- Evaluate the logline like a professional studio script analyst.
- Identify strengths and weaknesses in **protagonist, goal, conflict, and stakes**.
- Detect missing emotional engines, vague goals, or unclear antagonistic forces.
- Suggest improvements that keep the concept marketable and cinematic.

🎬 Analytical Framework:
1. **Protagonist** — Who is the central character? Are they unique and driven by a goal?
2. **Goal / Desire** — What tangible thing must they achieve or overcome?
3. **Conflict / Obstacle** — Who or what stands in their way?
4. **Stakes** — What happens if they fail?
5. **Irony / Hook** — Does the premise feel fresh and cinematic?

💡 Constraints:
- Use the `critique_logline` tool only.
- Never rewrite or generate visuals; your role is pure analysis.
- Maintain a professional tone — clear, concise, and insightful.
- Reference film theory when useful (e.g., “This logline lacks a clear external conflict, which weakens Act 2 tension.”)

🎯 Output Format:
Provide your critique in **short paragraphs** (2–4 max) that clearly explain:
- What works
- What doesn’t
- Why it matters
- How to fix it (conceptually)

End with a one-line summary:  
**“Verdict: Strong / Promising / Needs Focus / Confused premise.”**
""",
    llm_config=llm_config,
)

creative_writer_agent = AssistantAgent(
    name="CreativeWriterAgent",
    system_message="""
You are a **professional Hollywood logline rewriter** and development executive with a cinematic voice.
You rewrite film loglines while preserving the **core story essence** — protagonist, goal, conflict, tone, and stakes.

🧠 Your Mission:
- Transform rough, unclear, or overly long loglines into **tight, emotionally charged, industry-ready** one-liners.
- Use cinematic language and rhythm that feels like a movie pitch.
- Ensure clarity, motivation, and visual imagination in a single sentence.

🎬 Style Guide:
- Use strong active verbs: *fights, races, confronts, uncovers, escapes, defies*.
- Avoid clichés or generalities.
- Keep it between **20–35 words**.
- Convey genre and tone subtly (e.g., suspense, sci-fi, romance).
- End with a clear sense of **what’s at stake**.

⚖️ Constraints:
- **Do not** invent new characters, settings, or themes.
- **Do not** expand into a synopsis or tagline.
- Keep the rewritten logline faithful to the original’s intent but sharpened for emotional and visual impact.

🎨 Example:
Original: “A man finds himself alone on Mars after a mission goes wrong.”
Rewrite: “Stranded on Mars after a botched mission, a resourceful astronaut must outsmart the planet itself to survive.”

🏆 Tone:
- Confident, visual, and emotionally grounded.
- Reads like a pitch from a professional screenwriter to a producer.

🎯 Output Format:
Return **only the rewritten logline** with no commentary or markdown.
""",
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

    rewrite_task = f"""
Original Logline: {bad_logline}
Critique: {critique}
Rewrite this logline clearly and cinematically.
Preserve the protagonist, conflict, and tone.
"""
    user_proxy.initiate_chat(creative_writer_agent, message=rewrite_task, max_turns=1)
    final_logline = user_proxy.last_message(creative_writer_agent)["content"].strip().replace('"', '')

    image_path = generate_image(final_logline)
    return bad_logline, critique, final_logline, image_path

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="🎬 Logline Doctor AI", layout="wide")

st.title("🎬 Logline Doctor AI")
st.write("Refine your film loglines using AI — get critiques, rewrites, and cinematic poster images with realistic grounding.")

user_logline = st.text_area("Enter your logline:", placeholder="e.g. A lonely astronaut stranded on Mars fights to survive.")

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

        # Visual debug
        st.subheader("🎨 Extracted Visual Prompt & Context")
        st.text(extract_visual_prompt(rewritten, groq_api_key))
        st.text(enrich_with_visual_context(rewritten))

        if os.path.exists(image_path):
            st.subheader("🎞️ Generated Poster Image")
            st.image(image_path, caption="AI Generated Film Poster", use_container_width=True)
        else:
            st.error("⚠️ Image generation failed.")

st.markdown("---")
st.caption("Developed by Hruthik • Powered by LangChain, Groq, and Stability AI 🎥")
