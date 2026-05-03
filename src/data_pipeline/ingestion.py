import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.ingestion import IngestionPipeline

load_dotenv()

def get_env_variable(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"CRITICAL ERROR: '{name}' not found. Check your .env file.")
    return value

def ingest_curriculum_to_pinecone(data_dir: str, course_id: str):
    print(f"Loading documents from: {data_dir}")
    
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"WARNING: The directory {data_dir} is empty or missing.")
        return

    # 1. INITIALIZE LLAMAPARSE
    # This converts complex PDF/PPTX layouts into clean markdown
    parser = LlamaParse(
        api_key=get_env_variable("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
        verbose=True
    )
    
    # Map the parser to specific file types
    file_extractor = {".pdf": parser, ".pptx": parser}
    
    documents = SimpleDirectoryReader(
        data_dir, 
        file_extractor=file_extractor
    ).load_data()
    
    print(f"Successfully parsed {len(documents)} document pages/slides.")

    # 2. SET STABLE 2026 MODELS
    Settings.llm = GoogleGenAI(model="models/gemini-2.5-flash")
    embed_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-001")
    Settings.embed_model = embed_model

    # 3. CONFIGURE SPLITTER
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)

    # 4. INITIALIZE PINECONE
    print("Connecting to Pinecone...")
    api_key = get_env_variable("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    index_name = "tutor-agent-index"

    # 5. DIMENSION CHECK & INDEX CREATION (3072 Dimensions)
    if index_name in pc.list_indexes().names():
        existing_index = pc.describe_index(index_name)
        if existing_index.dimension != 3072:
            print(f"Upgrading index from {existing_index.dimension} to 3072 dimensions...")
            pc.delete_index(index_name)

    if index_name not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: '{index_name}' at 3072 dimensions...")
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine", 
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    pinecone_index = pc.Index(index_name)

    # 6. CONFIGURE NAMESPACE
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
    print(f"SUCCESS! Curriculum is now firmly grounded in namespace '{course_id}'.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    CURRICULUM_PATH = os.path.join(base_dir, "data")
    COURSE_NAMESPACE = "probability-and-stats-101"
    
    if not os.path.exists(CURRICULUM_PATH):
        os.makedirs(CURRICULUM_PATH)
        print(f"Created missing directory: {CURRICULUM_PATH}")
    else:
        try:
            ingest_curriculum_to_pinecone(CURRICULUM_PATH, COURSE_NAMESPACE)
        except Exception as e:
            print(f"An error occurred during ingestion:\n{e}")