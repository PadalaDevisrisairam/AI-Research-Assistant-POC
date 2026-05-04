from langchain_community.document_loaders import PyPDFLoader
import os 
from dotenv import load_dotenv 
from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import CharacterTextSplitter 
from langchain_community.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings
def load_rag():
    # Load PDF instead of txt
    loader = PyPDFLoader("./AIML-NOTE__6th-SEM.pdf")
    documents = loader.load()

    # Split into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

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

