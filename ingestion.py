# ingestion.py
from langchain_community.vectorstores.tiledb import VECTOR_INDEX_NAME
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
import os
from dotenv import load_dotenv
import certifi

load_dotenv()
ca = certifi.where()

MONGO_URI = os.getenv("MONGO_URI",
                      "mongodb+srv://esthertuyindula01:j5mfVplWtwKHLqdk@cluster0.nbocnv6.mongodb.net/?retryWrites=true&w=majority&appName=cluster0")  # Fallback for local

DB_NAME = "sample_mflix"
COLLECTION_NAME = "embedded_movies"
VECTOR_SEARCH_INDEX_NAME = "vector_rag_index"  # Ensure this matches your Atlas index

# Embedding Model (for Google Generative AI)
EMBEDDING_MODEL_NAME = "models/embedding-001"  # numDimensions must be 768 in Atlas index

# Gemini LLM Model
LLM_MODEL_NAME = "gemini2.0-flash"

# Gemini API Configuration - Canvas handles this, but explicitly passed to Langchain
GEMINI_API_KEY = "AIzaSyBNZsqvSrqUc1wFQ_dkqWhlWFaqR0ymfUI"  # This will be automatically populated by the Canvas environment


def ingest_data(file_path: list):
    client = MongoClient(MONGO_URI, tlsCAFile=ca)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Load documents
    # loader = PyPDFLoader(file_path)
    # documents = loader.load()
    sample_documents_data = [
        {"text": "The capital of France is Paris. Paris is known for the Eiffel Tower."},
        {"text": "The Amazon rainforest is the largest tropical rainforest in the world."},
        {"text": "Python is a popular high-level, interpreted programming language."},
        {"text": "MongoDB is a NoSQL document database that provides high performance, high availability, and easy scalability."},
        {"text": "Retrieval Augmented Generation (RAG) combines information retrieval with text generation to produce more accurate and informed responses."},
        {"text": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was built in 1889."},
        {"text": "NoSQL databases are non-relational databases, meaning they do not use the tabular schema of rows and columns found in traditional relational databases."},
        {"text": "Generative AI models, like large language models (LLMs), can create new content such as text, images, and code."},
        {"text": "The primary goal of RAG is to ground the LLM's responses in factual, external knowledge, reducing hallucinations."},
        {"text": "The global average temperature has been rising steadily due to climate change."},
        {"text": "Renewable energy sources include solar, wind, hydro, geothermal, and biomass."},
        {"text": "The human brain has approximately 86 billion neurons."},
        {"text": "Artificial intelligence (AI) is a broad field of computer science that aims to create intelligent machines."},
        {"text": "Machine learning is a subset of AI that enables systems to learn from data without being explicitly programmed."},
        {"text": "Deep learning is a subset of machine learning that uses neural networks with multiple layers."},
        {"text": "The internet was first developed in the late 1960s as ARPANET."},
        {"text": "World War II was a global war that lasted from 1939 to 1945."},
        {"text": "The first moon landing was achieved by Apollo 11 in 1969."},
        {"text": "Quantum computing uses principles of quantum mechanics to solve complex problems faster than classical computers."},
        {"text": "Blockchain technology is a decentralized, distributed ledger that records transactions across many computers."}
    ]

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(sample_documents_data)

    # Generate embeddings using GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

    # Store in MongoDB Atlas Vector Search
    MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection=collection,
        index_name=VECTOR_SEARCH_INDEX_NAME
    )
    print(f"Ingested {len(chunks)} chunks into MongoDB using Gemini embeddings.")
    client.close()

if __name__ == "__main__":
    # Example usage:
    # Make sure you have a 'data' directory with your PDFs, e.g., 'data/my_document.pdf'
    # ingest_data("data/your_document.pdf")
    print("Run `ingestion.py` with a file path to ingest data.")