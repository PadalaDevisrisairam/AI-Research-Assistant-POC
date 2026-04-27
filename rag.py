import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def load_rag():
    # Load document
    loader = TextLoader("data.txt")
    documents = loader.load()

    # Split into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Embeddings — runs locally, no API key needed
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},   # change to "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True}
    )

    # Store in vector DB
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")
    print("✅ Vector store saved!")
    return db



# # Test a query
# query = "Your question here"
# results = db.similarity_search(query, k=3)
# for doc in results:
#     print(doc.page_content)
#     print("---")

