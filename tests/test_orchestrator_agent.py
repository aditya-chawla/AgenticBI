"""
End-to-end tests for OrchestratorAgent.

Requires:
  - Docker Postgres (AdventureWorks) running on localhost:5432
  - Ollama with llama3 model running
  - ChromaDB pre-built via schema_ingestion_agent.py

Each test is a real pipeline run: NL2SQL → SQL Execution → Visualization.
Expect ~2-3 min per test (LLM inference is ~60s per call).
"""

import sys
import os

# Add agents to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'agents'))

from orchestrator_agent import OrchestratorAgent
from config import get_logger

logger = get_logger("test.orchestrator")


# ---------------------------------------------------------
# TEST 1: Simple aggregation → bar chart
# ---------------------------------------------------------
def test_simple_aggregation():
    """Top 5 employees by vacation hours — expects bar/table."""
    question = "Show me the top 5 employees by vacation hours."
    logger.info("TEST 1: %s", question)

    agent = OrchestratorAgent()
    result = agent.run(question)

    assert result["success"], f"Expected success, got error: {result['error']}"
    assert result["df"] is not None, "Expected DataFrame"
    assert len(result["df"]) > 0, "Expected non-empty DataFrame"
    assert result["sql"] is not None, "Expected SQL query"

    logger.info("  SQL:  %s", result["sql"][:80])
    logger.info("  Rows: %d", len(result["df"]))
    logger.info("  Viz:  %s", "OK" if not result["viz_failed"] else f"FAILED: {result['viz_error']}")

    return result


# ---------------------------------------------------------
# TEST 2: JOIN query → should trigger schema retrieval
# ---------------------------------------------------------
def test_join_query():
    """Products with inventory > 50 — requires Production.Product JOIN ProductInventory."""
    question = "Show product names and their total inventory quantities for products with inventory greater than 50."
    logger.info("TEST 2: %s", question)

    agent = OrchestratorAgent()
    result = agent.run(question)

    assert result["success"], f"Expected success, got error: {result['error']}"
    assert result["df"] is not None, "Expected DataFrame"
    assert len(result["df"]) > 0, "Expected non-empty DataFrame"

    logger.info("  SQL:  %s", result["sql"][:80])
    logger.info("  Rows: %d", len(result["df"]))
    logger.info("  Viz:  %s", "OK" if not result["viz_failed"] else f"FAILED: {result['viz_error']}")

    return result


# ---------------------------------------------------------
# TEST 3: Aggregation with GROUP BY → pie/bar chart
# ---------------------------------------------------------
def test_grouped_aggregation():
    """Sales breakdown — expects grouped results."""
    question = "Show total sales amount by territory region."
    logger.info("TEST 3: %s", question)

    agent = OrchestratorAgent()
    result = agent.run(question)

    assert result["success"], f"Expected success, got error: {result['error']}"
    assert result["df"] is not None, "Expected DataFrame"

    logger.info("  SQL:  %s", result["sql"][:80])
    logger.info("  Rows: %d", len(result["df"]))
    logger.info("  Viz:  %s", "OK" if not result["viz_failed"] else f"FAILED: {result['viz_error']}")

    return result


# ---------------------------------------------------------
# RUNNER
# ---------------------------------------------------------
if __name__ == "__main__":
    tests = [
        ("Simple Aggregation", test_simple_aggregation),
        ("JOIN Query", test_join_query),
        ("Grouped Aggregation", test_grouped_aggregation),
    ]

    passed = 0
    failed = 0
    results_summary = []

    for name, test_fn in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {name}")
        print(f"{'='*60}")
        try:
            result = test_fn()
            passed += 1
            status = "✅ PASSED"
            if result.get("viz_failed"):
                status += " (viz failed — partial success)"
            results_summary.append((name, status))

            # Print data preview
            if result["df"] is not None:
                print(f"\n📝 Data Preview:\n{result['df'].head().to_markdown()}")
            if result["chart_spec"]:
                print(f"📊 Chart: {result['chart_spec'].get('chart_type')} — {result['chart_spec'].get('title')}")

        except Exception as e:
            failed += 1
            results_summary.append((name, f"❌ FAILED: {e}"))
            logger.error("Test failed: %s — %s", name, e)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("📊 ORCHESTRATOR TEST SUMMARY")
    print(f"{'='*60}")
    for name, status in results_summary:
        print(f"   {name}: {status}")
    print(f"\n   {passed}/{passed + failed} tests passed")
