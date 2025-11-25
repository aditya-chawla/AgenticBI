from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_DB_PATH = "../src/agents/chroma_db_data"

def test_memory(query):
    print(f"\n❓ Testing Query: '{query}'")
    
    # Load the Database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)

    # Search for top 2 relevant tables
    results = vector_store.similarity_search(query, k=2)

    print("   Found these tables:")
    for doc in results:
        print(f"   - {doc.metadata['table_name']}")

if __name__ == "__main__":
    test_memory("Show me all the employees.")
    test_memory("Which products are currently in inventory?")