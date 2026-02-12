import sys
import os

# Add agents to path so we can import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'agents'))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import VECTOR_DB_PATH, EMBEDDING_MODEL, get_logger

logger = get_logger("test.schema_ingestion")

def test_memory(query):
    logger.info("Testing query: '%s'", query)
    
    # Load the Database
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)

    # Search for top 2 relevant tables
    results = vector_store.similarity_search(query, k=2)

    logger.info("Found these tables:")
    for doc in results:
        logger.info("   - %s", doc.metadata['table_name'])

if __name__ == "__main__":
    test_memory("Show me all the employees.")
    test_memory("Which products are currently in inventory?")