# app.py
import certifi
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

from ingestion import ingest_data

load_dotenv()
ca = certifi.where()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes by default. For production, specify origins.

# MongoDB connection details
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


# Global variables for RAG components
client = None
db = None
collection = None
vector_store = None
llm = None
embeddings = None
rag_chain = None


def initialize_rag_components():
    global client, db, collection, vector_store, llm, embeddings, rag_chain
    try:
        # Use synchronous MongoClient for Flask
        client = MongoClient(MONGO_URI, tlsCAFile=ca)
        client.admin.command('ping')  # Test connection
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        print("Connected to MongoDB!")

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, google_api_key=GEMINI_API_KEY)

        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name=VECTOR_SEARCH_INDEX_NAME
        )
        retriever = vector_store.as_retriever()
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )
        print("RAG components initialized successfully.")

        document_to_ingest =[
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
        if not document_to_ingest:
            print(f"Attempting to ingest data from: {document_to_ingest}")
            ingest_data(document_to_ingest)
        else:
            print(
                f"Warning: Document for ingestion not found at {document_to_ingest}. Please ensure it exists or ingest manually.")

    except Exception as e:
        print(f"Error during RAG component initialization: {e}")
        # In a production app, you might log this error and gracefully degrade
        raise SystemExit(f"Failed to initialize RAG components: {e}")


# Initialize components on application start
with app.app_context():
    initialize_rag_components()


# Route to serve the HTML file
@app.route("/home")
def index():
    return render_template("index.html")


# RAG query endpoint
@app.route("/rag-query", methods=["POST"])
def perform_rag_query():
    if not rag_chain:
        return jsonify({"detail": "RAG system not initialized."}), 500

    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"detail": "Query parameter is missing."}), 400

    try:
        print(f"Received query: {query}")
        # LangChain chains can be invoked synchronously in Flask routes
        # If your chain has async components, you would need to run Flask with an ASGI server
        # and ensure your LangChain components are truly async-compatible.
        # For simplicity here, we're assuming sync execution of the chain.
        response = rag_chain.invoke(query)
        print(f"Generated response: {response}")
        return jsonify({"answer": response})
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"detail": f"Error processing query: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)