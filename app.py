import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# 1. Load environment variables (Make sure GOOGLE_API_KEY is in your .env file)
load_dotenv()

def setup_gemini_rag():
    print("Initializing Gemini API...")
    
    # 2. Configure the LLM (The "Brain")
    # Gemini 1.5 Flash is excellent for high-speed, cost-effective RAG
    llm = Gemini(model="models/gemini-1.5-flash")
    
    # 3. Configure the Embedding Model (The "Search Engine")
    # text-embedding-004 is Google's free embedding model
    embed_model = GeminiEmbedding(model_name="models/text-embedding-004")
    
    # 4. Apply to LlamaIndex Global Settings
    # This tells the entire app to use Gemini instead of the OpenAI default
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    print("Gemini settings applied successfully.\n")

def run_agent():
    # 5. Load your documents
    # Note: Create a folder named "data" in the same directory and drop some PDFs/text files in it
    print("Loading documents from the 'data' folder...")
    try:
        documents = SimpleDirectoryReader("data").load_data()
    except ValueError:
        print("Error: Could not find a 'data' folder. Please create one and add some files.")
        return
        
    print(f"Loaded {len(documents)} document chunks. Building index...")
    
    # 6. Build the Vector Index
    index = VectorStoreIndex.from_documents(documents)
    
    # 7. Create the Query Engine
    query_engine = index.as_query_engine()
    
    print("\n==================================================")
    print("RAG Agent Ready! Ask a question (type 'quit' to exit)")
    print("==================================================")
    
    # 8. Start the chat loop
    while True:
        user_query = input("\nYour question: ")
        
        if user_query.lower() in ['quit', 'exit']:
            print("Exiting RAG agent. Goodbye!")
            break
            
        # Query the documents
        response = query_engine.query(user_query)
        print(f"\nAgent: {response}")

if __name__ == "__main__":
    # Make sure the API key is actually loaded
    if not os.environ.get("GOOGLE_API_KEY"):
        print("CRITICAL ERROR: GOOGLE_API_KEY not found in environment variables.")
        print("Please check your .env file.")
    else:
        setup_gemini_rag()
        run_agent()