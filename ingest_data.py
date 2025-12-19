import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
# --- NEW IMPORTS ---
from langchain_huggingface import HuggingFaceEmbeddings
# --- (End New Imports) ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Load environment variables from .env file
load_dotenv()

# Define the persistent directory for Chroma
CHROMA_DB_DIR = "./chroma_db"
DATA_FILE = "logline_principles.txt"

def ingest_data():
    """
    Loads data, splits it, creates embeddings, and stores them in ChromaDB
    using a free, local HuggingFace model.
    """
    
    # No API key check needed here anymore
    print(f"Loading data from {DATA_FILE}...")
    loader = TextLoader(DATA_FILE)
    documents = loader.load()

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    print("Creating embeddings using local HuggingFace model...")
    # This uses a popular, fast, and free model.
    # The first time you run this, it will download the model (a few hundred MB).
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Ingesting into ChromaDB...")
    # This creates a folder named CHROMA_DB_DIR
    db = Chroma.from_documents(
        texts, 
        embeddings, 
        persist_directory=CHROMA_DB_DIR
    )
    
    print(f"✅ Successfully ingested data and saved to {CHROMA_DB_DIR}")

if __name__ == "__main__":
    ingest_data()