import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline

# 1. LOAD ENVIRONMENT VARIABLES SAFELY
load_dotenv()

def get_env_variable(name):
    """Safely fetch environment variables and crash loudly if they are missing."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"CRITICAL ERROR: '{name}' not found. Check your .env file.")
    return value

def ingest_curriculum_to_pinecone(data_dir: str, course_id: str):
    """
    Ingests PDFs/PPTXs, chunks them safely, and uploads to a specific Pinecone namespace.
    """
    print(f"Loading documents from: {data_dir}")
    
    # Check if directory exists and has files
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"WARNING: The directory {data_dir} is empty or missing.")
        print("Please place at least one PDF or PPTX inside it and run again.")
        return

    documents = SimpleDirectoryReader(data_dir).load_data()
    print(f"Loaded {len(documents)} document pages/slides.")

    # 2. INITIALIZE MODERN GEMINI EMBEDDING
    # We use the 2026 standard 'gemini-embedding-001' which outputs 3072 dimensions.
    # It automatically looks for GOOGLE_API_KEY in the environment.
    print("Initializing Gemini Embedding Model...")
    embed_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-001")

    # 3. CONFIGURE A FREE-TIER FRIENDLY SPLITTER
    # chunk_size=512 is perfect for a tutor agent. It grabs about half a slide at a time.
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)

    # 4. INITIALIZE PINECONE
    print("Connecting to Pinecone...")
    api_key = get_env_variable("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    index_name = "tutor-agent-index"

    # 5. DIMENSION MISMATCH CHECK & INDEX CREATION
    # If the index exists but is stuck at the old 768 size, delete it to make room.
    if index_name in pc.list_indexes().names():
        existing_index = pc.describe_index(index_name)
        if existing_index.dimension != 3072:
            print(f"Mismatch detected! Upgrading index from {existing_index.dimension} to 3072 dimensions...")
            pc.delete_index(index_name)

    # Create the new 3072-dimension index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: '{index_name}' at 3072 dimensions...")
        pc.create_index(
            name=index_name,
            dimension=3072, # Must exactly match the Gemini model's output
            metric="cosine", 
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    pinecone_index = pc.Index(index_name)

    # 6. CONFIGURE NAMESPACE
    # The course_id (e.g., probability-and-stats-101) ensures data stays isolated.
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index, 
        namespace=course_id
    )

    # 7. RUN THE PIPELINE
    print("Running ingestion pipeline. Embedding and uploading to Pinecone...")
    pipeline = IngestionPipeline(
        transformations=[
            splitter,
            embed_model,
        ],
        vector_store=vector_store,
    )
    
    pipeline.run(documents=documents)
    print(f"\n✅ SUCCESS! Curriculum is now firmly grounded in namespace '{course_id}'.")


if __name__ == "__main__":
    # Dynamically find the absolute path to the data folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    CURRICULUM_PATH = os.path.join(base_dir, "data")
    
    # Isolate this specific batch of lectures
    COURSE_NAMESPACE = "probability-and-stats-101"
    
    # Create the data directory if it doesn't exist yet
    if not os.path.exists(CURRICULUM_PATH):
        os.makedirs(CURRICULUM_PATH)
        print(f"Created missing directory: {CURRICULUM_PATH}")
        print("Please drop your PDFs there and run this script again.")
    else:
        try:
            ingest_curriculum_to_pinecone(CURRICULUM_PATH, COURSE_NAMESPACE)
        except Exception as e:
            print(f"\n An error occurred during ingestion:\n{e}")