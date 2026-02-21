import psycopg2
import os
import shutil
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import DB_CONFIG, VECTOR_DB_PATH, EMBEDDING_MODEL, get_logger

logger = get_logger("agent.schema_ingestion")

class SchemaIngestionAgent:
    def __init__(self, db_config):
        self.db_config = db_config

    def extract_ddl(self):
        """
        Connects to DB and generates 'CREATE TABLE' statements.
        Returns a list of LangChain Documents.
        """
        logger.info(f"Connecting to database '{self.db_config['dbname']}'...")
        docs = []
        conn = None
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # 1. Get relevant tables (Skipping system stuff)
            cur.execute("""
                SELECT table_schema, table_name 
                FROM information_schema.tables 
                WHERE table_schema IN ('Person', 'Sales', 'Production', 'Purchasing', 'HumanResources') 
                AND table_type = 'BASE TABLE';
            """)
            tables = cur.fetchall()
            logger.info(f"Found {len(tables)} tables. Generating DDL blueprints...")

            for schema, table in tables:
                full_table_name = f"{schema}.{table}"
                
                # 2. Get columns
                cur.execute(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_schema = '{schema}' AND table_name = '{table}';
                """)
                columns = cur.fetchall()
                
                # 3. Construct DDL (The 'Context' for the AI)
                ddl_lines = [f"CREATE TABLE {full_table_name} ("]
                for col in columns:
                    col_name, dtype, nullable = col
                    null_str = "NULL" if nullable == "YES" else "NOT NULL"
                    ddl_lines.append(f"  {col_name} {dtype} {null_str},")
                ddl_lines.append(");")
                
                ddl_content = "\n".join(ddl_lines)

                # 4. Create Document object
                doc = Document(
                    page_content=ddl_content,
                    metadata={
                        "table_name": full_table_name,
                        "schema": schema
                    }
                )
                docs.append(doc)

            return docs

        except Exception as e:
            logger.error(f"DB Error during DDL extraction: {e}")
            return []
        finally:
            if conn: conn.close()

    def build_index(self, documents):
        if not documents:
            logger.warning("No documents to index. Skipping.")
            return

        logger.info(f"Vectorizing {len(documents)} tables using '{EMBEDDING_MODEL}'...")
        
        # 1. Clear old index if it exists (so we don't get duplicates)
        if os.path.exists(VECTOR_DB_PATH):
            shutil.rmtree(VECTOR_DB_PATH)

        # 2. Initialize Free Local Embedding Model
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # 3. Create Vector DB
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        
        logger.info(f"Schema index saved to '{VECTOR_DB_PATH}'")

if __name__ == "__main__":
    agent = SchemaIngestionAgent(DB_CONFIG)
    schema_docs = agent.extract_ddl()
    agent.build_index(schema_docs)