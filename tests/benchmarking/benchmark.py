"""
Agentic BI — Chapter 7 Benchmark Suite
=======================================
Runs the 10 E2E queries from Chapter 6 through the full pipeline,
measures SESR, VSR, EESR, Correction Loop Effectiveness, and latency.

Usage:
    cd src/agents
    python ../../tests/benchmark.py [--runs 3] [--disable-graph]

Requirements:
    - Docker Postgres (AdventureWorks) running
    - OPENROUTER_API_KEY set in .env
    - ChromaDB + schema_graph.json pre-built via schema_ingestion_agent.py

Output:
    - benchmark_results.json  (raw per-query-per-run data)
    - benchmark_summary.json  (aggregate SESR, VSR, EESR, latency stats)
    - Prints formatted tables to stdout for easy copy into the report
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime

# --- Path setup (run from src/agents or project root) ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENTS_DIR = os.path.join(os.path.dirname(_THIS_DIR), "src", "agents")
if _AGENTS_DIR not in sys.path:
    sys.path.insert(0, _AGENTS_DIR)

from orchestrator_agent import OrchestratorAgent
from config import get_logger, VECTOR_DB_PATH

logger = get_logger("benchmark")

# ============================================================
# THE 10 E2E QUERIES (from Chapter 6)
# ============================================================
QUERIES = [
    {
        "id": 1,
        "query": "Show me the top 10 best-selling products by total revenue.",
        "tests": "SUM() aggregation, multi-table JOIN, GROUP BY, ORDER BY DESC, LIMIT",
        "expected_viz": "Bar Chart",
    },
    {
        "id": 2,
        "query": "What is the total sales revenue by year and month?",
        "tests": "Date parsing/extraction, time-series grouping",
        "expected_viz": "Line Chart",
    },
    {
        "id": 3,
        "query": "Show me the distribution of employees by marital status.",
        "tests": "Simple COUNT() aggregation with single-column GROUP BY",
        "expected_viz": "Pie/Donut Chart",
    },
    {
        "id": 4,
        "query": "What is the average sick leave hours by job title?",
        "tests": "Categorical grouping with AVG()",
        "expected_viz": "Horizontal Bar Chart",
    },
    {
        "id": 5,
        "query": "Show me total sales amount by sales territory country.",
        "tests": "Cross-schema JOIN between sales and territory tables",
        "expected_viz": "Bar Chart",
    },
    {
        "id": 6,
        "query": "Which top 5 vendors do we spend the most money with?",
        "tests": "Purchasing schema JOIN (PurchaseOrderHeader + Vendor)",
        "expected_viz": "Bar Chart",
    },
    {
        "id": 7,
        "query": "Show me the total inventory quantity grouped by product subcategory.",
        "tests": "Multi-table JOIN across ProductInventory, Product, ProductSubcategory",
        "expected_viz": "Horizontal Bar Chart",
    },
    {
        "id": 8,
        "query": "Compare the standard cost versus the list price for all products.",
        "tests": "Retrieval of two continuous numeric variables without aggregation",
        "expected_viz": "Scatter Plot",
    },
    {
        "id": 9,
        "query": "Show me the number of orders placed online versus in-store.",
        "tests": "Boolean flag (OnlineOrderFlag) for grouping and counting",
        "expected_viz": "Pie/Bar Chart",
    },
    {
        "id": 10,
        "query": "Who are the top 5 sales representatives by total sales, and what were their total sales?",
        "tests": "Complex conditional JOINs spanning Sales + Person schemas with ID-to-name resolution",
        "expected_viz": "Bar Chart",
    },
]


def run_single_query(agent, query_text):
    """
    Runs one query through the orchestrator. Returns a result dict with
    all the fields we need for metrics.
    """
    start = time.time()
    try:
        result = agent.run(query_text)
    except Exception as e:
        elapsed = time.time() - start
        return {
            "sql_generated": False,
            "sql_success": False,
            "rows_returned": 0,
            "viz_success": False,
            "viz_failed": False,
            "correction_loops": 0,
            "error": str(e),
            "sql": None,
            "chart_type": None,
            "elapsed_seconds": round(elapsed, 2),
        }

    elapsed = time.time() - start

    sql_generated = result.get("sql") is not None and len(result.get("sql", "")) > 0
    sql_success = result.get("success", False) or result.get("viz_failed", False)
    # If success=True (even partial), SQL worked
    rows = len(result["df"]) if result.get("df") is not None else 0
    viz_success = result.get("success", False) and result.get("figure") is not None
    viz_failed = result.get("viz_failed", False)

    # Estimate correction loops from the final state
    # The orchestrator doesn't directly expose retry_count in the result,
    # so we infer: if SQL has a correction hint pattern or error was recovered
    # We check if the result mentions retries in the logs — but simpler:
    # the orchestrator logs this, so for the benchmark we can check the
    # result dict. For now, we track whether the query needed retries
    # by looking at whether sql_success required error recovery.
    # A more precise way: patch the orchestrator to return retry_count.
    # For now, mark as 0 (successful first try) or infer from elapsed time.

    chart_type = None
    if result.get("chart_spec"):
        chart_type = result["chart_spec"].get("chart_type")

    retry_count = result.get("retry_count", 0)

    return {
        "sql_generated": sql_generated,
        "sql_success": sql_success,
        "rows_returned": rows,
        "viz_success": viz_success,
        "viz_failed": viz_failed,
        "correction_loops": retry_count,
        "error": result.get("error"),
        "sql": result.get("sql"),
        "chart_type": chart_type,
        "elapsed_seconds": round(elapsed, 2),
    }


def run_benchmark(num_runs=3, disable_graph=False):
    """
    Main benchmark loop. Runs all queries `num_runs` times.
    """
    # --- Optionally disable GraphRAG ---
    graph_path = os.path.join(VECTOR_DB_PATH, "schema_graph.json")
    dict_path = os.path.join(VECTOR_DB_PATH, "schema_dict.json")
    graph_backup = graph_path + ".bak"
    dict_backup = dict_path + ".bak"

    if disable_graph:
        logger.info("=" * 60)
        logger.info("DISABLING GRAPH (vanilla RAG mode)")
        logger.info("=" * 60)
        if os.path.exists(graph_path):
            os.rename(graph_path, graph_backup)
        if os.path.exists(dict_path):
            os.rename(dict_path, dict_backup)

    try:
        agent = OrchestratorAgent()
        all_results = []

        for q in QUERIES:
            query_results = []
            for run_num in range(1, num_runs + 1):
                print(f"\n{'='*60}")
                print(f"  Query {q['id']}/{len(QUERIES)}  |  Run {run_num}/{num_runs}")
                print(f"  \"{q['query'][:60]}...\"")
                print(f"{'='*60}")

                r = run_single_query(agent, q["query"])
                r["query_id"] = q["id"]
                r["run"] = run_num
                r["query_text"] = q["query"]
                query_results.append(r)

                status = "✅" if r["viz_success"] else ("⚠️ viz fail" if r["sql_success"] else "❌")
                print(f"  {status}  SQL={'✓' if r['sql_success'] else '✗'}  "
                      f"Viz={'✓' if r['viz_success'] else '✗'}  "
                      f"Rows={r['rows_returned']}  "
                      f"Time={r['elapsed_seconds']}s  "
                      f"Chart={r['chart_type'] or 'N/A'}")

            all_results.extend(query_results)

        return all_results

    finally:
        # --- Restore graph files if disabled ---
        if disable_graph:
            if os.path.exists(graph_backup):
                os.rename(graph_backup, graph_path)
            if os.path.exists(dict_backup):
                os.rename(dict_backup, dict_path)
            logger.info("Graph files restored.")


def compute_metrics(results, num_runs):
    """
    Computes SESR, VSR, EESR, Correction Loop Effectiveness, and latency stats.
    """
    total_runs = len(results)
    num_queries = total_runs // num_runs

    # --- Per-query aggregation ---
    per_query = {}
    for r in results:
        qid = r["query_id"]
        if qid not in per_query:
            per_query[qid] = {
                "query_text": r["query_text"],
                "runs": [],
                "sql_successes": 0,
                "viz_successes": 0,
                "e2e_successes": 0,
                "total_runs": 0,
                "latencies": [],
                "retried_runs": 0,
                "recovered_runs": 0,
            }
        pq = per_query[qid]
        pq["runs"].append(r)
        pq["total_runs"] += 1
        if r["sql_success"] and r["rows_returned"] > 0:
            pq["sql_successes"] += 1
        if r["viz_success"]:
            pq["viz_successes"] += 1
        if r["viz_success"]:
            pq["e2e_successes"] += 1
        pq["latencies"].append(r["elapsed_seconds"])
        # Track correction loop effectiveness
        if r["correction_loops"] > 0:
            pq["retried_runs"] += 1
            if r["sql_success"] and r["rows_returned"] > 0:
                pq["recovered_runs"] += 1

    # --- Aggregate metrics ---
    total_sql_success = sum(pq["sql_successes"] for pq in per_query.values())
    total_viz_success = sum(pq["viz_successes"] for pq in per_query.values())
    total_e2e_success = sum(pq["e2e_successes"] for pq in per_query.values())

    sesr = (total_sql_success / total_runs) * 100 if total_runs > 0 else 0
    vsr = (total_viz_success / total_sql_success) * 100 if total_sql_success > 0 else 0
    eesr = (total_e2e_success / total_runs) * 100 if total_runs > 0 else 0

    # Correction Loop Effectiveness: % of initially-failing queries recovered by retries
    total_retried = sum(pq["retried_runs"] for pq in per_query.values())
    total_recovered = sum(pq["recovered_runs"] for pq in per_query.values())
    cle = (total_recovered / total_retried) * 100 if total_retried > 0 else 0

    all_latencies = [r["elapsed_seconds"] for r in results if r["viz_success"]]
    avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
    min_latency = min(all_latencies) if all_latencies else 0
    max_latency = max(all_latencies) if all_latencies else 0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": num_queries,
        "runs_per_query": num_runs,
        "total_runs": total_runs,
        "SESR": round(sesr, 1),
        "VSR": round(vsr, 1),
        "EESR": round(eesr, 1),
        "CLE": round(cle, 1),
        "total_retried": total_retried,
        "total_recovered": total_recovered,
        "latency_avg_seconds": round(avg_latency, 1),
        "latency_min_seconds": round(min_latency, 1),
        "latency_max_seconds": round(max_latency, 1),
        "per_query": {},
    }

    for qid, pq in sorted(per_query.items()):
        avg_lat = sum(pq["latencies"]) / len(pq["latencies"]) if pq["latencies"] else 0
        summary["per_query"][qid] = {
            "query": pq["query_text"][:60],
            "sql_success_rate": f"{pq['sql_successes']}/{pq['total_runs']}",
            "viz_success_rate": f"{pq['viz_successes']}/{pq['total_runs']}",
            "e2e_success_rate": f"{pq['e2e_successes']}/{pq['total_runs']}",
            "avg_latency": round(avg_lat, 1),
            "retried": pq["retried_runs"],
            "recovered": pq["recovered_runs"],
        }

    return summary


def print_report_table(summary):
    """
    Prints formatted tables ready to paste into the report.
    """
    print("\n")
    print("=" * 90)
    print("  TABLE 1: END-TO-END SUCCESS RATES PER QUERY")
    print("=" * 90)
    print(f"  {'Q#':<4} {'SQL':>8} {'Viz':>8} {'E2E':>8} {'Avg Time':>10}  {'Query':<50}")
    print(f"  {'--':<4} {'---':>8} {'---':>8} {'---':>8} {'--------':>10}  {'-----':<50}")

    for qid, pq in sorted(summary["per_query"].items()):
        print(f"  {qid:<4} {pq['sql_success_rate']:>8} {pq['viz_success_rate']:>8} "
              f"{pq['e2e_success_rate']:>8} {pq['avg_latency']:>8.1f}s  {pq['query']:<50}")

    print()
    print("=" * 90)
    print("  AGGREGATE METRICS")
    print("=" * 90)
    print(f"  SQL Execution Success Rate (SESR):  {summary['SESR']}%")
    print(f"  Visualization Success Rate (VSR):   {summary['VSR']}%")
    print(f"  End-to-End Success Rate (EESR):     {summary['EESR']}%")
    print(f"  Correction Loop Effectiveness (CLE):{summary['CLE']}%  ({summary['total_recovered']}/{summary['total_retried']} recovered)")
    print(f"  Average Latency (successful):       {summary['latency_avg_seconds']}s")
    print(f"  Min / Max Latency:                  {summary['latency_min_seconds']}s / {summary['latency_max_seconds']}s")
    print(f"  Total Runs:                         {summary['total_runs']}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Agentic BI Benchmark Suite")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per query (default: 3)")
    parser.add_argument("--disable-graph", action="store_true",
                        help="Disable GraphRAG (vanilla RAG baseline)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for output JSON files")
    args = parser.parse_args()

    mode = "vanilla_rag" if args.disable_graph else "graphrag"
    print(f"\n{'#'*60}")
    print(f"  AGENTIC BI BENCHMARK — Mode: {mode.upper()}")
    print(f"  {args.runs} runs per query, {len(QUERIES)} queries")
    print(f"  Estimated time: {args.runs * len(QUERIES) * 20 // 60}-{args.runs * len(QUERIES) * 35 // 60} minutes")
    print(f"{'#'*60}\n")

    results = run_benchmark(num_runs=args.runs, disable_graph=args.disable_graph)
    summary = compute_metrics(results, args.runs)

    # --- Save raw results ---
    results_file = os.path.join(args.output_dir, f"benchmark_results_{mode}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to: {results_file}")

    # --- Save summary ---
    summary_file = os.path.join(args.output_dir, f"benchmark_summary_{mode}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")

    # --- Print report-ready table ---
    print_report_table(summary)


if __name__ == "__main__":
    main()