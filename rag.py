import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()


DATA_FILE = "data.txt"
INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings():
    """
    Create and return the Hugging Face embedding model.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )


def load_rag():
    """
    Load an existing FAISS index if available.

    If the index does not exist:
    1. Load data.txt
    2. Split the document
    3. Generate embeddings
    4. Create FAISS index
    5. Save it locally
    """

    # Create embedding model once for this function call
    embeddings = create_embeddings()

    # ---------------------------------
    # CASE 1: Existing index found
    # ---------------------------------

    if os.path.exists(INDEX_PATH):

        print("📂 Loading existing FAISS vector store...")

        db = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        print("✅ Existing vector store loaded!")

        return db

    # ---------------------------------
    # CASE 2: Index doesn't exist
    # ---------------------------------

    print("🔨 FAISS index not found. Creating new vector store...")

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Could not find '{DATA_FILE}'. "
            "Make sure it exists in the project directory."
        )

    # Load document
    loader = TextLoader(
        DATA_FILE,
        encoding="utf-8"
    )

    documents = loader.load()

    # Split documents intelligently
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    print(f"📄 Created {len(docs)} document chunks.")

    # Create FAISS vector database
    db = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )

    # Save locally
    db.save_local(INDEX_PATH)

    print("✅ Vector store created and saved!")

    return db