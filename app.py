import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables from the .env file to keep API keys secure.
load_dotenv()

# Configure the global settings for LlamaIndex.
# We use gpt-4o-mini for a balance of cost and reasoning capability.
# Temperature is set to 0.2 to keep the agent's responses factual and grounded in the text,
# which is crucial for an educational tool.
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Define the embedding model. This translates our text into numerical vectors.
# text-embedding-3-small is currently the standard for efficient, high-quality retrieval.
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

def main():
    # Initialize LlamaParse. 
    # Standard PDF parsers often destroy the formatting of lecture slides.
    # LlamaParse uses vision models to understand the layout and returns structured Markdown,
    # preserving hierarchies like titles, bullet points, and tables.
    parser = LlamaParse(
        result_type="markdown", 
        verbose=True
    )
    
    # Map the parser to handle any .pdf files encountered by the directory reader.
    file_extractor = {".pdf": parser}
    
    print("Loading documents into memory...")
    
    # SimpleDirectoryReader scans the target folder and applies the appropriate parser.
    # It loads the raw Markdown text into Document objects.
    documents = SimpleDirectoryReader(
        "./data", 
        file_extractor=file_extractor
    ).load_data()

    print("Generating embeddings and building the vector index...")
    
    # VectorStoreIndex chunks the Document objects into smaller segments,
    # passes them to the embedding model, and stores the resulting vectors in memory.
    # In a later phase, we will replace this in-memory storage with a persistent database like ChromaDB.
    index = VectorStoreIndex.from_documents(documents)

    # Convert the vector index into a query engine. 
    # This engine handles taking a user's text query, embedding it, finding the most similar 
    # document chunks, and passing those chunks to the LLM to generate a final answer.
    query_engine = index.as_query_engine()
    
    print("System ready. Type 'exit' to quit.")
    
    # Begin the interactive chat loop.
    while True:
        user_query = input("\nStudent: ")
        if user_query.lower() in ["exit", "quit"]:
            break
            
        # The query engine executes the retrieval and generation steps synchronously.
        response = query_engine.query(user_query)
        print(f"\nAgent: {response}")

if __name__ == "__main__":
    main()