import psycopg2
import pandas as pd
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

from config import DB_CONFIG, LLM_MODEL, get_logger

logger = get_logger("agent.sql_execution")


# ---------------------------------------------------------
# DETERMINISTIC QUOTE FIXER
# ---------------------------------------------------------
# AdventureWorks uses case-sensitive "Schema"."Table" identifiers.
# The LLM frequently generates unquoted or partially-quoted names.
# This regex-based fixer is a safety net that runs BEFORE execution.

import re as _re

# Known AdventureWorks schemas
_AW_SCHEMAS = (
    "HumanResources", "Person", "Production", "Purchasing",
    "Sales", "dbo", "pr",
)
_SCHEMA_PATTERN = "|".join(_AW_SCHEMAS)

# Matches Schema.Table (with optional existing quotes) and ensures both parts
# are double-quoted.  Handles:
#   HumanResources.Employee          →  "HumanResources"."Employee"
#   "HumanResources".Employee        →  "HumanResources"."Employee"
#   "HumanResources"."Employee"      →  "HumanResources"."Employee"  (no-op)
_SCHEMA_TABLE_RE = _re.compile(
    r'"?(' + _SCHEMA_PATTERN + r')"?\s*\.\s*"?([A-Za-z_]\w*)"?',
    _re.IGNORECASE,
)



# Matches alias.ColumnName patterns where alias is a short lowercase identifier
# and ColumnName starts with an uppercase letter (CamelCase).
# Does NOT match if the column is already quoted: alias."Column"
# Pattern 1: unquoted alias   →  e.BusinessEntityID  →  e."BusinessEntityID"
_ALIAS_COLUMN_UNQUOTED_RE = _re.compile(
    r'\b([a-z]\w{0,4})\.(?!")([A-Z]\w*)\b'
)
# Pattern 2: quoted alias     →  "s".TerritoryID     →  "s"."TerritoryID"
_ALIAS_COLUMN_QUOTED_RE = _re.compile(
    r'"([a-z]\w{0,4})"\.(?!")([A-Z]\w*)\b'
)


def _ensure_quoted_identifiers(sql: str) -> str:
    """Deterministically double-quote all Schema.Table and alias.Column references."""
    # Step 1: fix Schema.Table
    def _schema_replacer(m):
        schema = m.group(1)
        table = m.group(2)
        return f'"{ schema }"."{ table }"'
    sql = _SCHEMA_TABLE_RE.sub(_schema_replacer, sql)

    # Step 2a: fix unquoted alias.Column  (e.BusinessEntityID → e."BusinessEntityID")
    def _col_unquoted(m):
        return f'{ m.group(1) }."{ m.group(2) }"'
    sql = _ALIAS_COLUMN_UNQUOTED_RE.sub(_col_unquoted, sql)

    # Step 2b: fix quoted alias.Column  ("s".TerritoryID → "s"."TerritoryID")
    def _col_quoted(m):
        return f'"{ m.group(1) }"."{m.group(2)}"'
    sql = _ALIAS_COLUMN_QUOTED_RE.sub(_col_quoted, sql)

    return sql

# ---------------------------------------------------------
# 1. DEFINE THE STATE
#    (The memory passed between the Execute Node and the Fix Node)
# ---------------------------------------------------------
class ExecutionState(TypedDict):
    sql_query: str          # The query we are trying to run
    result_data: Optional[str] # Markdown string of results (for display)
    result_dict: Optional[dict] # Raw data as dict (for Visualization Agent)
    error_message: Optional[str] # DB Error if failed
    retry_count: int        # How many times have we tried?

# ---------------------------------------------------------
# 2. DEFINE THE NODES
# ---------------------------------------------------------

def execute_sql_node(state: ExecutionState):
    """
    Attempts to run the SQL in Postgres.
    Applies deterministic quote-fixing before execution.
    """
    raw_query = state['sql_query']
    query = _ensure_quoted_identifiers(raw_query)
    current_retries = state.get('retry_count', 0)

    if query != raw_query:
        logger.info("Quote-fixer applied to SQL identifiers")
        logger.debug("Before: %s", raw_query[:120])
        logger.debug("After:  %s", query[:120])
    
    logger.info(f"Executing SQL (attempt {current_retries + 1}): {query[:80]}...")

    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        # Using pandas to easily get column names and formatting
        df = pd.read_sql(query, conn)
        
        # If successful, clear errors and save data
        logger.info(f"SQL executed successfully — {len(df)} rows returned")
        return {
            "result_data": df.to_markdown(),
            "result_dict": df.to_dict(orient="records"),
            "error_message": None,
            "retry_count": current_retries
        }

    except Exception as e:
        error_msg = str(e)
        logger.error(f"SQL execution failed: {error_msg}")
        return {
            "error_message": error_msg,
            "result_data": None,
            "result_dict": None,
            "retry_count": current_retries
        }
    finally:
        if conn: conn.close()

def fix_query_node(state: ExecutionState):
    """
    Uses Llama 3 to fix the query based on the Postgres error message.
    """
    broken_query = state['sql_query']
    error_msg = state['error_message']
    retries = state['retry_count']
    
    logger.info(f"Calling LLM to fix query (retry {retries + 1}/3)...")
    logger.debug(f"Error was: {error_msg[:150]}")

    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    
    template = """
    You are a PostgreSQL Expert. 
    A previous SQL query failed to execute. Fix it based on the error message.

    ### BROKEN QUERY:
    {broken_query}

    ### POSTGRES ERROR:
    {error_msg}

    ### INSTRUCTIONS:
    1. Output ONLY the fixed SQL query.
    2. Do NOT output markdown markers.
    3. Fix syntax errors.
    4. CRITICAL: This database is Case-Sensitive.
       - You MUST wrap ALL Table and Schema names in Double Quotes.
       - Example: Use "Production"."Product" instead of Production.Product.

    ### FIXED SQL:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    fixed_sql = chain.invoke({
        "broken_query": broken_query,
        "error_msg": error_msg
    })
    
    # --- IMPROVED CLEANER ---
    # 1. Remove Markdown
    clean_sql = fixed_sql.replace("```sql", "").replace("```", "").strip()
    
    # 2. Regex to find the actual SQL statement
    # Try with trailing semicolon first, then without
    match = re.search(r"(SELECT\b.*?;)", clean_sql, re.DOTALL | re.IGNORECASE)
    if not match:
        # No semicolon — grab from first SELECT to end of string
        match = re.search(r"(SELECT\b.*)", clean_sql, re.DOTALL | re.IGNORECASE)
    if match:
        clean_sql = match.group(1).rstrip(";").strip()
        
    logger.info(f"LLM suggested fix: {clean_sql[:80]}...")
    
    return {
        "sql_query": clean_sql,
        "retry_count": retries + 1
    }
# ---------------------------------------------------------
# 3. DEFINE THE ROUTER LOGIC
# ---------------------------------------------------------
def should_continue(state: ExecutionState):
    """
    Decides if we are done or need to loop back to fix.
    """
    if state['result_data'] is not None:
        return "success" # We got data!
    
    if state['retry_count'] >= 3:
        logger.warning("Max retries (3) reached. Giving up.")
        return "give_up"
    
    return "retry" # Loop back to fix_node

# ---------------------------------------------------------
# 4. BUILD THE GRAPH
# ---------------------------------------------------------
def build_execution_graph():
    workflow = StateGraph(ExecutionState)

    # Add Nodes
    workflow.add_node("execute", execute_sql_node)
    workflow.add_node("fix", fix_query_node)

    # Entry Point
    workflow.set_entry_point("execute")

    # Conditional Edges
    workflow.add_conditional_edges(
        "execute",
        should_continue,
        {
            "success": END,
            "give_up": END,
            "retry": "fix"
        }
    )

    # Connect Fix back to Execute (The Loop)
    workflow.add_edge("fix", "execute")

    return workflow.compile()

# ---------------------------------------------------------
# MAIN EXECUTION API
# ---------------------------------------------------------
class SQLExecutor:
    def __init__(self):
        self.app = build_execution_graph()

    def run(self, sql_query: str):
        """
        Takes an SQL query, runs it, fixes it if needed.
        Returns the final DataFrame (markdown) or Error.
        """
        initial_state = {
            "sql_query": sql_query,
            "retry_count": 0,
            "result_data": None,
            "result_dict": None,
            "error_message": None
        }
        
        # Invoke the LangGraph workflow
        final_state = self.app.invoke(initial_state)
        
        if final_state['result_data']:
            # Return DataFrame + markdown string
            df = pd.DataFrame(final_state['result_dict'])
            markdown = final_state['result_data']
            return True, df, markdown
        else:
            return False, None, final_state['error_message']

# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------
if __name__ == "__main__":
    executor = SQLExecutor()
    
    print("="*50)
    print("🧪 TEST 1: Broken SQL (Checking Self-Correction)")
    print("="*50)
    # Missing FROM clause on purpose to trigger the Fix Node
    bad_sql = "SELECT Name Production.Product WHERE ProductID = 1;"
    
    success, df, output = executor.run(bad_sql)
    if success:
        print(f"\n📝 Final Result:\n{output}\n")
    else:
        print(f"\n❌ Error:\n{output}\n")


    print("="*50)
    print("🧪 TEST 2: Generated SQL (Checking Happy Path)")
    print("="*50)
    # Corrected SQL with Double Quotes
    generated_sql = """
    SELECT p."Name"
    FROM "Production"."Product" p
    JOIN "Production"."ProductInventory" pi ON p."ProductID" = pi."ProductID"
    WHERE pi."Quantity" > 50;
    """
    
    success, df, output = executor.run(generated_sql)
    
    if success:
        print(f"\n📝 Final Result (DataFrame shape: {df.shape}):")
        print(df.head(10).to_markdown())
    else:
        print(f"\n❌ Error:\n{output}")