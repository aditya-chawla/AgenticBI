"""
Orchestrator Agent — wires NL2SQL → SQLExecutor → VisualizationAgent
into a LangGraph pipeline with inter-agent correction loops.

Correction Loop 1 (SQL errors):
    SQLExecutor fails → error fed back to NL2SQL as correction_hint → regenerate

Correction Loop 2 (empty results):
    SQL returns 0 rows → hint to NL2SQL to try a different approach

Partial success (Option A):
    If SQL succeeds but visualization fails → success=True, figure=None,
    viz_failed=True with the error surfaced.
"""

import pandas as pd
from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, END

from nl2sql_agent import NL2SQLAgent
from sql_execution_agent import SQLExecutor
from visualization_agent import VisualizationAgent
from config import get_logger

logger = get_logger("agent.orchestrator")

# Max orchestrator-level retries (on top of each agent's internal retries)
MAX_RETRIES = 2

# Module-level singletons — HuggingFace model loaded once, not per call
_nl2sql_agent: Optional[Any] = None
_sql_executor: Optional[Any] = None


def _get_nl2sql_agent() -> "NL2SQLAgent":
    global _nl2sql_agent
    if _nl2sql_agent is None:
        _nl2sql_agent = NL2SQLAgent()
    return _nl2sql_agent


def _get_sql_executor() -> "SQLExecutor":
    global _sql_executor
    if _sql_executor is None:
        _sql_executor = SQLExecutor()
    return _sql_executor


# ---------------------------------------------------------
# 1. ORCHESTRATOR STATE
# ---------------------------------------------------------
class OrchestratorState(TypedDict):
    # --- Input ---
    user_question: str

    # --- NL2SQL outputs ---
    sql_query: Optional[str]
    correction_hint: Optional[str]       # Fed back on retry

    # --- SQL Execution outputs ---
    sql_success: Optional[bool]
    result_dict: Optional[list]          # df.to_dict("records") — serialisable
    markdown: Optional[str]

    # --- Visualization outputs ---
    viz_success: Optional[bool]
    figure_json: Optional[str]           # fig.to_json()
    chart_spec: Optional[dict]

    # --- Control flow ---
    error_message: Optional[str]
    retry_count: int
    stage: str                           # current pipeline stage


# ---------------------------------------------------------
# 2. NODE FUNCTIONS
# ---------------------------------------------------------
def generate_sql_node(state: OrchestratorState) -> dict:
    """Call NL2SQL agent to produce a SQL query."""
    logger.info("=" * 60)
    logger.info("STAGE: generate_sql  (retry %d/%d)", state["retry_count"], MAX_RETRIES)
    logger.info("=" * 60)

    agent = _get_nl2sql_agent()
    hint = state.get("correction_hint")

    try:
        sql = agent.generate_sql(state["user_question"], correction_hint=hint)
        logger.info("SQL generated (%d chars)", len(sql))
        logger.debug("SQL:\n%s", sql)
        return {
            "sql_query": sql,
            "error_message": None,
            "stage": "generate_sql",
        }
    except Exception as e:
        error = f"NL2SQL failed: {e}"
        logger.error(error)
        return {
            "sql_query": None,
            "error_message": error,
            "stage": "generate_sql",
        }


def execute_sql_node(state: OrchestratorState) -> dict:
    """Call SQLExecutor to run the query against Postgres."""
    logger.info("=" * 60)
    logger.info("STAGE: execute_sql")
    logger.info("=" * 60)

    executor = _get_sql_executor()

    try:
        success, df, output = executor.run(state["sql_query"])
    except Exception as e:
        error = f"SQLExecutor crashed: {e}"
        logger.error(error)
        return {
            "sql_success": False,
            "result_dict": None,
            "markdown": None,
            "error_message": error,
            "stage": "execute_sql",
        }

    if success:
        row_count = len(df)
        logger.info("SQL executed — %d rows returned", row_count)
        return {
            "sql_success": True,
            "result_dict": df.to_dict("records"),
            "markdown": output,
            "error_message": None,
            "stage": "execute_sql",
        }
    else:
        logger.warning("SQL execution failed: %s", output)
        return {
            "sql_success": False,
            "result_dict": None,
            "markdown": None,
            "error_message": output,
            "stage": "execute_sql",
        }


def check_execution_node(state: OrchestratorState) -> dict:
    """
    Router node — decides what happens after SQL execution:
      • success + rows  → continue to visualize
      • success + 0 rows → correction loop (empty results)
      • failure          → correction loop (SQL error)
      • retries exhausted → give up
    """
    if state["sql_success"] and state.get("result_dict"):
        row_count = len(state["result_dict"])
        if row_count > 0:
            logger.info("Check passed — %d rows, proceeding to visualization", row_count)
            return {}  # No state changes; routing handled by conditional edge

    # If we get here, something needs correction
    retry = state["retry_count"] + 1

    if not state["sql_success"]:
        hint = f"SQL execution error: {state.get('error_message', 'unknown')}"
        logger.warning("Correction loop — SQL error (retry %d/%d)", retry, MAX_RETRIES)
    else:
        hint = (
            "The query returned 0 rows. The SQL ran successfully but produced "
            "no results. Try different table joins, filters, or column names."
        )
        logger.warning("Correction loop — empty results (retry %d/%d)", retry, MAX_RETRIES)

    return {
        "correction_hint": hint,
        "retry_count": retry,
    }


def visualize_node(state: OrchestratorState) -> dict:
    """Call VisualizationAgent to produce a Plotly chart."""
    logger.info("=" * 60)
    logger.info("STAGE: visualize")
    logger.info("=" * 60)

    df = pd.DataFrame(state["result_dict"])
    agent = VisualizationAgent()

    try:
        success, fig, spec_or_error = agent.run(df, state["user_question"])
    except Exception as e:
        error = f"VisualizationAgent crashed: {e}"
        logger.error(error)
        return {
            "viz_success": False,
            "figure_json": None,
            "chart_spec": None,
            "error_message": error,
            "stage": "visualize",
        }

    if success:
        logger.info("Visualization succeeded — chart type: %s", spec_or_error.get("chart_type"))
        return {
            "viz_success": True,
            "figure_json": fig.to_json(),
            "chart_spec": spec_or_error,
            "error_message": None,
            "stage": "visualize",
        }
    else:
        logger.warning("Visualization failed: %s", spec_or_error)
        return {
            "viz_success": False,
            "figure_json": None,
            "chart_spec": None,
            "error_message": spec_or_error,
            "stage": "visualize",
        }


# ---------------------------------------------------------
# 3. ROUTING FUNCTIONS
# ---------------------------------------------------------
def route_after_generate(state: OrchestratorState) -> str:
    """After NL2SQL: if we got SQL → execute; else → give up."""
    if state.get("sql_query"):
        return "execute_sql"
    logger.error("NL2SQL produced no SQL — aborting")
    return "give_up"


def route_after_check(state: OrchestratorState) -> str:
    """
    After check_execution:
      • SQL success + rows  → visualize
      • retries exhausted   → give up
      • else                → retry generate_sql
    """
    if state["sql_success"] and state.get("result_dict") and len(state["result_dict"]) > 0:
        return "visualize"
    if state["retry_count"] >= MAX_RETRIES:
        logger.error("Max orchestrator retries exhausted (%d/%d)", state["retry_count"], MAX_RETRIES)
        return "give_up"
    return "retry_generate"


def route_after_visualize(state: OrchestratorState) -> str:
    """After visualization: always proceed to END (partial success handled in .run())."""
    return "done"


# ---------------------------------------------------------
# 4. BUILD THE GRAPH
# ---------------------------------------------------------
def _build_graph() -> StateGraph:
    workflow = StateGraph(OrchestratorState)

    # Nodes
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("check_execution", check_execution_node)
    workflow.add_node("visualize", visualize_node)

    # Entry
    workflow.set_entry_point("generate_sql")

    # Edges
    workflow.add_conditional_edges("generate_sql", route_after_generate, {
        "execute_sql": "execute_sql",
        "give_up": END,
    })

    workflow.add_edge("execute_sql", "check_execution")

    workflow.add_conditional_edges("check_execution", route_after_check, {
        "visualize": "visualize",
        "retry_generate": "generate_sql",
        "give_up": END,
    })

    workflow.add_conditional_edges("visualize", route_after_visualize, {
        "done": END,
    })

    return workflow


# ---------------------------------------------------------
# 5. ORCHESTRATOR WRAPPER CLASS
# ---------------------------------------------------------
class OrchestratorAgent:
    """
    High-level entry point.

    Usage:
        agent = OrchestratorAgent()
        result = agent.run("Show total sales by region")

    Returns a dict:
        {
            "success": bool,
            "sql": str | None,
            "df": pd.DataFrame | None,
            "markdown": str | None,
            "figure": plotly.Figure | None,
            "chart_spec": dict | None,
            "viz_failed": bool,
            "viz_error": str | None,
            "error": str | None,
        }
    """

    def __init__(self):
        self.graph = _build_graph().compile()

    def run(self, question: str) -> dict:
        logger.info("=" * 70)
        logger.info("ORCHESTRATOR START — %s", question[:100])
        logger.info("=" * 70)

        initial_state: OrchestratorState = {
            "user_question": question,
            "sql_query": None,
            "correction_hint": None,
            "sql_success": None,
            "result_dict": None,
            "markdown": None,
            "viz_success": None,
            "figure_json": None,
            "chart_spec": None,
            "error_message": None,
            "retry_count": 0,
            "stage": "init",
        }

        final = self.graph.invoke(initial_state)

        # --- Build result dict ---
        result = {
            "success": False,
            "sql": final.get("sql_query"),
            "df": None,
            "markdown": final.get("markdown"),
            "figure": None,
            "chart_spec": final.get("chart_spec"),
            "viz_failed": False,
            "viz_error": None,
            "error": final.get("error_message"),
        }

        # Reconstruct DataFrame if we have data
        if final.get("result_dict"):
            result["df"] = pd.DataFrame(final["result_dict"])

        # Determine success level
        if final.get("sql_success") and final.get("result_dict"):
            if final.get("viz_success"):
                # Full success — SQL + viz both worked
                import plotly.io as pio
                result["success"] = True
                result["figure"] = pio.from_json(final["figure_json"])
                logger.info("ORCHESTRATOR COMPLETE — full success")
            else:
                # Partial success — SQL ok but viz failed (Option A)
                result["success"] = True
                result["viz_failed"] = True
                result["viz_error"] = final.get("error_message")
                logger.warning("ORCHESTRATOR COMPLETE — partial success (viz failed)")
        else:
            logger.error("ORCHESTRATOR COMPLETE — failed: %s", final.get("error_message"))

        return result


# ---------------------------------------------------------
# 6. STANDALONE TEST
# ---------------------------------------------------------
if __name__ == "__main__":
    agent = OrchestratorAgent()

    question = "Show me the top 5 employees by vacation hours."
    print(f"\n{'='*70}")
    print(f"🧪 ORCHESTRATOR TEST: {question}")
    print(f"{'='*70}\n")

    result = agent.run(question)

    print(f"\n{'='*70}")
    print("📊 RESULT SUMMARY")
    print(f"{'='*70}")
    print(f"  Success:    {result['success']}")
    print(f"  SQL:        {result['sql'][:80] if result['sql'] else 'None'}...")
    print(f"  DataFrame:  {result['df'].shape if result['df'] is not None else 'None'}")
    print(f"  Markdown:   {'Yes' if result['markdown'] else 'No'}")
    print(f"  Figure:     {'Yes' if result['figure'] else 'No'}")
    print(f"  Chart Spec: {result['chart_spec']}")
    print(f"  Viz Failed: {result['viz_failed']}")
    print(f"  Viz Error:  {result['viz_error']}")
    print(f"  Error:      {result['error']}")

    if result["df"] is not None:
        print(f"\n📝 Data Preview:\n{result['df'].head().to_markdown()}")

    # Show chart in browser if we have a figure
    if result.get("figure") is not None:
        result["figure"].show()
