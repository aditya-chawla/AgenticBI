import psycopg2
import os
import shutil
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import networkx as nx
import json

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
        schema_graph = nx.DiGraph()
        schema_dict = {}
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
                
                # --- Append Foreign Keys ---
                cur.execute(f"""
                    SELECT
                        kcu.column_name,
                        ccu.table_schema AS foreign_table_schema,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM 
                        information_schema.table_constraints AS tc 
                        JOIN information_schema.key_column_usage AS kcu
                          ON tc.constraint_name = kcu.constraint_name
                          AND tc.table_schema = kcu.table_schema
                        JOIN information_schema.constraint_column_usage AS ccu
                          ON ccu.constraint_name = tc.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY' 
                        AND tc.table_schema = '{schema}'
                        AND tc.table_name = '{table}';
                """)
                fks = cur.fetchall()
                if fks:
                    ddl_lines.append("\n-- Foreign Keys --")
                    for fk in fks:
                        col_name_fk, f_schema, f_table, f_col = fk
                        ddl_lines.append(f"-- {col_name_fk} references {f_schema}.{f_table}({f_col})")

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

                # 5. Populate Graph and Dictionary
                schema_graph.add_node(full_table_name, schema=schema)
                schema_dict[full_table_name] = ddl_content
                
                if fks:
                    for fk in fks:
                        col_name_fk, f_schema, f_table, f_col = fk
                        foreign_table = f"{f_schema}.{f_table}"
                        # Directed Edge from Table A -> Table B (A references B)
                        schema_graph.add_edge(full_table_name, foreign_table, key=col_name_fk)

            return docs, schema_graph, schema_dict

        except Exception as e:
            logger.error(f"DB Error during DDL extraction: {e}")
            return [], nx.DiGraph(), {}
        finally:
            if conn: conn.close()

    def build_index(self, documents, schema_graph, schema_dict):
        if not documents:
            logger.warning("No documents to index. Skipping.")
            return

        logger.info(f"Vectorizing {len(documents)} tables using '{EMBEDDING_MODEL}'...")
        
        # 1. Clear old index if it exists (so we don't get duplicates)
        if os.path.exists(VECTOR_DB_PATH):
            for item in os.listdir(VECTOR_DB_PATH):
                item_path = os.path.join(VECTOR_DB_PATH, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

        # 2. Initialize Free Local Embedding Model
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # 3. Create Vector DB
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        
        # 4. Save Graph Topology and Dictionary
        with open(os.path.join(VECTOR_DB_PATH, "schema_graph.json"), "w") as f:
            json.dump(nx.node_link_data(schema_graph), f)
            
        with open(os.path.join(VECTOR_DB_PATH, "schema_dict.json"), "w") as f:
            json.dump(schema_dict, f)
            
        logger.info(f"Schema Vector index, Graph topology, and Dictionary saved to '{VECTOR_DB_PATH}'")

if __name__ == "__main__":
    agent = SchemaIngestionAgent(DB_CONFIG)
    schema_docs, schema_graph, schema_dict = agent.extract_ddl()
    agent.build_index(schema_docs, schema_graph, schema_dict)