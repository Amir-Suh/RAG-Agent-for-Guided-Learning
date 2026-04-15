import os
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

load_dotenv()

class GroundedRetriever:
    def __init__(self, course_id: str):
        # STABLE 2026 CONFIGURATION
        Settings.llm = GoogleGenAI(model="models/gemini-2.5-flash")
        Settings.embed_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-001")

        # DATABASE CONNECTION
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = "tutor-agent-index"
        pinecone_index = pc.Index(index_name)
        
        # NAMESPACE ISOLATION
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index, 
            namespace=course_id
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store, 
            storage_context=storage_context
        )
        
        # similarity_top_k=5 allows the agent to pull more context for summaries
        self.query_engine = self.index.as_query_engine(similarity_top_k=5)

    @retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(5))
    def ask(self, query: str):
        """Standard query method with exponential backoff to handle free tier limits."""
        return self.query_engine.query(query)


# --- TERMINAL TESTING LOOP ---
if __name__ == "__main__":
    COURSE = "probability-and-stats-101"
    print(f"--- Initializing Stable Retriever for: {COURSE} ---")
    
    try:
        tutor_tool = GroundedRetriever(COURSE)
        while True:
            user_query = input("\nAsk a question (or 'exit'): ")
            if user_query.lower() == 'exit': 
                break
            
            print("Synthesizing answer... (Handling rate limits if necessary)")
            response = tutor_tool.ask(user_query)
            print(f"\n[Grounded Answer]:\n{response}")
            
    except Exception as e:
        print(f"\nFinal Failure: {e}")