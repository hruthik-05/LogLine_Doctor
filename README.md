


# 🎬 LogLine Doctor

**LogLine Doctor** is an AI-powered application that analyzes, critiques, and rewrites film loglines using professional screenplay principles.  
It helps writers transform rough ideas into **clear, cinematic, industry-ready loglines**, and also generates a **movie-poster-style image** based on the final logline.

---

## ✨ Features

- 🧠 **Professional Logline Critique**
  - Evaluates loglines using core storytelling principles:
    - Protagonist
    - Goal
    - Conflict
    - Stakes

- ✍️ **Cinematic Logline Rewrite**
  - Rewrites loglines in a Hollywood pitch style
  - Preserves the original idea while improving clarity and emotional impact

- 📚 **RAG (Retrieval-Augmented Generation)**
  - Uses ChromaDB vector stores built from:
    - Logline principles
    - Visual descriptions
  - Retrieves relevant context for grounded analysis

- 🎨 **AI Movie Poster Generation**
  - Generates a cinematic poster-style image from the rewritten logline

- 🌐 **Multiple Interfaces**
  - Flask-based API backend
  - Streamlit-based web UI

---

## 🗂️ Project Structure

```text
LogLine_Doctor/
├── api_backend.py          # Flask API backend
├── streamlit_app.py        # Streamlit web application
├── final.py                # Core orchestration logic
├── img.py                  # Image generation logic
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── templates/              # HTML templates (Flask)
├── logline_principles.txt  # Storytelling principles (RAG source)
├── visual_descriptions.txt # Visual grounding data (RAG source)
├── .gitignore              # Git ignore rules
````

> ⚠️ **Note**
> Virtual environments (`env/`), vector databases, generated images, and `.env` files are intentionally **excluded from GitHub**.

---

## 🛠️ Tech Stack

* **Python**
* **Flask** – API backend
* **Streamlit** – Web UI
* **LangChain**
* **AutoGen**
* **Groq (LLaMA 3.1)**
* **ChromaDB**
* **Stability AI**

---

## 🚀 Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/hruthik-05/LogLine_Doctor.git
cd LogLine_Doctor
```

### 2️⃣ Create and activate a virtual environment

```bash
python3 -m venv env
source env/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Environment variables

Create a `.env` file (do **not** commit it):

```env
GROQ_API_KEY=your_groq_api_key
STABILITY_API_KEY=your_stability_api_key
```

---

## ▶️ Running the Project

### 🔹 Run Streamlit App

```bash
streamlit run streamlit_app.py
```

### 🔹 Run Flask API

```bash
python api_backend.py
```

---

## 🧠 How It Works

1. User submits a logline
2. Logline is analyzed using RAG + screenplay principles
3. AI provides structured critique
4. Logline is rewritten cinematically
5. Visual context is extracted
6. AI generates a movie-poster-style image

---

## 🔒 Security

* `.env` files are ignored via `.gitignore`
* API keys must **never** be committed
* Generated data (vector DBs, images) are rebuilt at runtime

---

## 📌 Future Improvements

* User authentication
* Save and compare multiple loglines
* Export critiques as PDF
* Cloud deployment
* Model fine-tuning for storytelling quality

---

## 👨‍💻 Author

**Hruthik**

---

🎥 *“A logline isn’t a summary — it’s a promise of the movie.”*

```

---

✅ **Single Markdown file**  
✅ **No separated parts**  
✅ **GitHub-ready**  
✅ **Copy–paste once and done**

If you want any change (shorter / hackathon / badges), say the word 👍
```
