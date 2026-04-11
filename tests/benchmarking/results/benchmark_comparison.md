# Benchmark Comparison: GraphRAG vs Vanilla RAG

**Date:** 2026-04-11 | **Runs per query:** 3 | **Queries:** 10 | **Total runs per mode:** 30

---

## Per-Query Results

| Q# | Query | GraphRAG E2E | Vanilla RAG E2E | GraphRAG Latency | Vanilla RAG Latency | GraphRAG Retries | Vanilla RAG Retries |
|----|-------|:------------:|:---------------:|:----------------:|:-------------------:|:----------------:|:-------------------:|
| 1  | Top 10 best-selling products by total revenue | 3/3 | 3/3 | 8.4s | 10.3s | 0 | 0 |
| 2  | Total sales revenue by year and month | 3/3 | 3/3 | 5.4s | 15.1s | 0 | 0 |
| 3  | Distribution of employees by marital status | 3/3 | 3/3 | 3.9s | 7.5s | 0 | 0 |
| 4  | Average sick leave hours by job title | 3/3 | 3/3 | 5.6s | 11.4s | 0 | 1 |
| 5  | Total sales amount by sales territory country | **2/3** | 1/3 | 51.0s | 12.0s | 2 | 3 |
| 6  | Top 5 vendors by spend | 3/3 | 3/3 | 4.9s | 12.6s | 0 | 0 |
| 7  | Total inventory by product subcategory | 0/3 | **2/3** | 23.6s | 28.8s | 3 | 2 |
| 8  | Standard cost vs list price for all products | 3/3 | 3/3 | 9.3s | 10.5s | 0 | 0 |
| 9  | Orders placed online vs in-store | **2/3** | 1/3 | 10.5s | 29.1s | 0 | 1 |
| 10 | Top 5 sales reps by total sales | **2/3** | 1/3 | 11.1s | 253.2s | 1 | 2 |

---

## Aggregate Metrics

| Metric | GraphRAG | Vanilla RAG | Delta |
|--------|:--------:|:-----------:|:-----:|
| **SESR** (SQL Execution Success Rate) | 83.3% | 83.3% | 0.0 pp |
| **VSR** (Visualization Success Rate) | 96.0% | 92.0% | +4.0 pp |
| **EESR** (End-to-End Success Rate) | **80.0%** | 76.7% | **+3.3 pp** |
| **CLE** (Correction Loop Effectiveness) | 16.7% (1/6) | 44.4% (4/9) | -27.7 pp |
| **Avg Latency** (successful queries) | **11.9s** | 12.4s | **-0.5s** |
| **Min Latency** | 3.2s | 5.0s | -1.8s |
| **Max Latency** | 134.7s | 39.3s | +95.4s |
| **Total Retries** | 6 | 9 | -3 |

---

## Key Takeaways

1. **GraphRAG achieves higher E2E success (80.0% vs 76.7%)** — it wins on queries 5, 9, and 10 (cross-schema JOINs and complex lookups), which are exactly the cases where graph-augmented schema retrieval provides better context.

2. **GraphRAG needs fewer retries (6 vs 9)** — better schema context means fewer SQL errors on the first attempt, reducing correction loop overhead.

3. **GraphRAG is faster on average (11.9s vs 12.4s)** — fewer retries translates to lower mean latency, though it has a higher max latency outlier (134.7s vs 39.3s) from Query 5's difficult territory joins.

4. **Vanilla RAG wins on Query 7 (inventory by subcategory)** — 2/3 vs 0/3, suggesting the graph may be routing to incorrect schema paths for this particular multi-table join pattern.

5. **CLE is higher for Vanilla RAG (44.4% vs 16.7%)** — but this is because Vanilla RAG needs more retries in the first place; GraphRAG avoids the errors rather than recovering from them.
