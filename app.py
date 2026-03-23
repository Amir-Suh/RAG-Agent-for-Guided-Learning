import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_parse import LlamaParse
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# Load environment variables 
load_dotenv()

def setup_gemini_rag():
    print("Initializing Gemini API...")
    
    # Configure the LLM and Embedding Model
    llm = Gemini(model="models/gemini-1.5-flash")
    embed_model = GeminiEmbedding(model_name="models/text-embedding-004")
    
    # Apply to LlamaIndex Global Settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    print("Gemini settings applied successfully.\n")

def run_agent():
    print("Initializing LlamaParse...")
    parser = LlamaParse(
        result_type="markdown",
        verbose=True
    )
    
    file_extractor = {".pdf": parser, ".pptx": parser}
    
    print("Loading lecture slides from the 'data' folder...")
    try:
        documents = SimpleDirectoryReader(
            "data", 
            file_extractor=file_extractor
        ).load_data()
    except ValueError:
        print("Error: Could not find a 'data' folder. Please create one and add your slides.")
        return
        
    print(f"Loaded {len(documents)} document pages/slides.")
    print("Extracting concepts and building the Knowledge Graph... (This will take a moment)")
    
    kg_extractor = SimpleLLMPathExtractor(
        llm=Settings.llm, 
        max_paths_per_chunk=10, 
        num_workers=4
    )
    
    index = PropertyGraphIndex.from_documents(
        documents,
        kg_extractors=[kg_extractor],
        show_progress=True
    )
    
    query_engine = index.as_query_engine(
        include_text=True 
    )
    
    print("\n==================================================")
    print("GraphRAG Tutor Ready! Ask a concept question (type 'quit' to exit)")
    print("==================================================")
    
    while True:
        user_query = input("\nYour question: ")
        
        if user_query.lower() in ['quit', 'exit']:
            print("Exiting Tutor. Goodbye!")
            break
            
        response = query_engine.query(user_query)
        print(f"\nTutor: {response}")

if __name__ == "__main__":
    missing_keys = []
    if not os.environ.get("GOOGLE_API_KEY"):
        missing_keys.append("GOOGLE_API_KEY")
    if not os.environ.get("LLAMA_CLOUD_API_KEY"):
        missing_keys.append("LLAMA_CLOUD_API_KEY")
        
    if missing_keys:
        print(f"CRITICAL ERROR: Missing API keys in .env file: {', '.join(missing_keys)}")
        print("Please update your .env file before running.")
    else:
        setup_gemini_rag()
        run_agent()