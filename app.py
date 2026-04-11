import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_parse import LlamaParse

# --- MODERN IMPORTS ---
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
# ----------------------

load_dotenv()

def setup_gemini_rag():
    print("Initializing Modern Gemini SDK...")
    
    # FIX: Removed 'models/' prefix for the new unified SDK
    llm = GoogleGenAI(model="gemini-2.5-flash")
    
    # FIX: Removed 'models/' prefix here as well
    embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001")
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    print("Gemini settings applied successfully.\n")

def run_agent():
    print("Initializing LlamaParse (Unified SDK)...")
    # Note: You can ignore the LlamaParse deprecation warning in the terminal!
    parser = LlamaParse(
        result_type="markdown",
        verbose=True
    )
    
    file_extractor = {".pdf": parser, ".pptx": parser}
    
    print("Loading lecture slides from the 'data' folder...")
    try:
        # We ensure the reader uses the new parser
        documents = SimpleDirectoryReader(
            "data", 
            file_extractor=file_extractor
        ).load_data()
    except Exception as e:
        print(f"Error loading documents: {e}")
        return
        
    print(f"Loaded {len(documents)} document pages. Building vector index...")
    
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True
    )
    
    query_engine = index.as_query_engine(include_text=True)
    
    print("\n==================================================")
    print("RAG Tutor Ready!")
    print("==================================================")
    
    while True:
        user_query = input("\nYour question: (type 'quit' to exit) ")
        if user_query.lower() in ['quit', 'exit']:
            print("Exiting Tutor. Goodbye!")
            break
            
        response = query_engine.query(user_query)
        print(f"\nTutor: {response}")

if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY") or not os.environ.get("LLAMA_CLOUD_API_KEY"):
        print("CRITICAL ERROR: Missing API keys in .env file.")
    else:
        setup_gemini_rag()
        run_agent()