import sys
import os
import pandas as pd
import numpy as np
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import tempfile

# Add agents to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'agents'))

from visualization_agent import VisualizationAgent


# ---------------------------------------------------------
# TEST 1: BAR CHART
# Data: Few categories with clear values to compare
# Trigger: "compare", "by category"
# ---------------------------------------------------------
def test_bar_chart():
    print("=" * 50)
    print("🧪 TEST 1: Bar Chart")
    print("=" * 50)
    agent = VisualizationAgent()
    df = pd.DataFrame({
        "Product": ["Widget A", "Widget B", "Widget C", "Widget D"],
        "Revenue": [45000, 32000, 58000, 21000],
        "Region": ["East", "West", "East", "West"],
    })
    success, fig, spec = agent.run(df, "Compare the revenue of each product")
    assert success, "Bar chart generation failed"
    print(f"   ✅ Chart type: {spec['chart_type']} | Title: {spec['title']}")
    return fig, spec


# ---------------------------------------------------------
# TEST 2: LINE CHART
# Data: Sequential/time-ordered x-axis with numeric y
# Trigger: "trend", "over time", date-like column names
# ---------------------------------------------------------
def test_line_chart():
    print("\n" + "=" * 50)
    print("🧪 TEST 2: Line Chart")
    print("=" * 50)
    agent = VisualizationAgent()
    df = pd.DataFrame({
        "Date": ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05",
                 "2025-06", "2025-07", "2025-08", "2025-09", "2025-10"],
        "MonthlySales": [120, 135, 148, 162, 155, 170, 182, 195, 188, 210],
    })
    success, fig, spec = agent.run(df, "Show me the trend of monthly sales over time")
    assert success, "Line chart generation failed"
    print(f"   ✅ Chart type: {spec['chart_type']} | Title: {spec['title']}")
    return fig, spec


# ---------------------------------------------------------
# TEST 3: SCATTER CHART
# Data: Two continuous numeric columns, many rows
# Trigger: "relationship", "correlation", "between X and Y"
# ---------------------------------------------------------
def test_scatter_chart():
    print("\n" + "=" * 50)
    print("🧪 TEST 3: Scatter Chart")
    print("=" * 50)
    agent = VisualizationAgent()
    np.random.seed(42)
    df = pd.DataFrame({
        "SquareFootage": np.random.randint(800, 3500, 50),
        "Price": np.random.randint(150000, 750000, 50),
        "Bedrooms": np.random.choice([2, 3, 4, 5], 50),
    })
    success, fig, spec = agent.run(df, "What is the correlation between square footage and price?")
    assert success, "Scatter chart generation failed"
    print(f"   ✅ Chart type: {spec['chart_type']} | Title: {spec['title']}")
    return fig, spec


# ---------------------------------------------------------
# TEST 4: PIE CHART
# Data: Small number of categories with values summing to a whole
# Trigger: "proportion", "share", "percentage", "what portion"
# ---------------------------------------------------------
def test_pie_chart():
    print("\n" + "=" * 50)
    print("🧪 TEST 4: Pie Chart")
    print("=" * 50)
    agent = VisualizationAgent()
    df = pd.DataFrame({
        "Region": ["North America", "Europe", "Asia", "Other"],
        "MarketShare": [42, 28, 22, 8],
    })
    success, fig, spec = agent.run(df, "What proportion of market share does each region hold?")
    assert success, "Pie chart generation failed"
    print(f"   ✅ Chart type: {spec['chart_type']} | Title: {spec['title']}")
    return fig, spec


# ---------------------------------------------------------
# TEST 5: HISTOGRAM
# Data: Single numeric column with many continuous values
# Trigger: "distribution of", "frequency"
# ---------------------------------------------------------
def test_histogram():
    print("\n" + "=" * 50)
    print("🧪 TEST 5: Histogram")
    print("=" * 50)
    agent = VisualizationAgent()
    np.random.seed(42)
    df = pd.DataFrame({
        "EmployeeSalary": np.random.normal(75000, 15000, 200).astype(int),
    })
    success, fig, spec = agent.run(df, "Show the frequency distribution of employee salaries")
    assert success, "Histogram generation failed"
    print(f"   ✅ Chart type: {spec['chart_type']} | Title: {spec['title']}")
    return fig, spec


# ---------------------------------------------------------
# TEST 6: BOX PLOT
# Data: Numeric values grouped by category
# Trigger: "spread", "range", "outliers", "quartiles"
# ---------------------------------------------------------
def test_box_chart():
    print("\n" + "=" * 50)
    print("🧪 TEST 6: Box Plot")
    print("=" * 50)
    agent = VisualizationAgent()
    np.random.seed(42)
    departments = ["Engineering"] * 30 + ["Sales"] * 30 + ["Marketing"] * 30
    salaries = list(np.random.normal(95000, 12000, 30).astype(int)) + \
               list(np.random.normal(72000, 18000, 30).astype(int)) + \
               list(np.random.normal(68000, 10000, 30).astype(int))
    df = pd.DataFrame({
        "Department": departments,
        "Salary": salaries,
    })
    success, fig, spec = agent.run(df, "Show the salary spread and outliers across departments")
    assert success, "Box chart generation failed"
    print(f"   ✅ Chart type: {spec['chart_type']} | Title: {spec['title']}")
    return fig, spec


# ---------------------------------------------------------
# TEST 7: HEATMAP
# Data: Two categorical axes + one numeric value (matrix-like)
# Trigger: "heatmap", "matrix", "cross-tabulation"
# ---------------------------------------------------------
def test_heatmap():
    print("\n" + "=" * 50)
    print("🧪 TEST 7: Heatmap")
    print("=" * 50)
    agent = VisualizationAgent()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    hours = ["9AM", "10AM", "11AM", "12PM", "1PM", "2PM", "3PM", "4PM"]
    rows = []
    np.random.seed(42)
    for day in days:
        for hour in hours:
            rows.append({"Day": day, "Hour": hour, "Visitors": np.random.randint(10, 200)})
    df = pd.DataFrame(rows)
    success, fig, spec = agent.run(df, "Show a heatmap of website visitors by day and hour")
    assert success, "Heatmap generation failed"
    print(f"   ✅ Chart type: {spec['chart_type']} | Title: {spec['title']}")
    return fig, spec


# ---------------------------------------------------------
# TEST 8: TREEMAP
# Data: Hierarchical categories with values
# Trigger: "hierarchical breakdown", "treemap"
# ---------------------------------------------------------
def test_treemap():
    print("\n" + "=" * 50)
    print("🧪 TEST 8: Treemap")
    print("=" * 50)
    agent = VisualizationAgent()
    df = pd.DataFrame({
        "Continent": ["Americas", "Americas", "Americas", "Europe", "Europe", "Europe", "Asia", "Asia"],
        "Country": ["USA", "Canada", "Brazil", "UK", "Germany", "France", "Japan", "India"],
        "GDP_Billion": [21000, 1700, 1400, 2800, 3800, 2700, 5000, 2900],
    })
    success, fig, spec = agent.run(df, "Show a treemap of GDP broken down by continent and country")
    assert success, "Treemap generation failed"
    print(f"   ✅ Chart type: {spec['chart_type']} | Title: {spec['title']}")
    return fig, spec


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def build_html(chart_results):
    """Build a single HTML page with all charts in a grid."""
    chart_divs = ""
    for name, status, fig, spec in chart_results:
        if fig:
            chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
            badge = f'<span style="background:#22c55e;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px">{spec["chart_type"]}</span>'
        else:
            chart_html = '<div style="display:flex;align-items:center;justify-content:center;height:300px;color:#ef4444;">❌ Generation Failed</div>'
            badge = '<span style="background:#ef4444;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px">FAILED</span>'

        chart_divs += f"""
        <div style="background:#1e1e2e;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.3)">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                <h3 style="margin:0;color:#e2e8f0">{name}</h3>
                {badge}
            </div>
            {chart_html}
        </div>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Agentic BI - Visualization Agent Test Results</title>
    <script src="https://cdn.plot.ly/plotly-3.3.1.min.js"></script>
    <style>
        body {{ background: #0f0f1a; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; }}
        h1 {{ text-align: center; color: #a78bfa; margin-bottom: 8px; }}
        .subtitle {{ text-align: center; color: #94a3b8; margin-bottom: 30px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1400px; margin: 0 auto; }}
        @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <h1>🎨 Visualization Agent - Test Results</h1>
    <p class="subtitle">All charts generated by LLM-driven Vizro pipeline</p>
    <div class="grid">
        {chart_divs}
    </div>
</body>
</html>"""


if __name__ == "__main__":
    tests = [
        ("Bar Chart", test_bar_chart),
        ("Line Chart", test_line_chart),
        ("Scatter Chart", test_scatter_chart),
        ("Pie Chart", test_pie_chart),
        ("Histogram", test_histogram),
        ("Box Plot", test_box_chart),
        ("Heatmap", test_heatmap),
        ("Treemap", test_treemap),
    ]

    passed = 0
    failed = 0
    results = []
    chart_results = []

    for name, test_fn in tests:
        try:
            fig, spec = test_fn()
            passed += 1
            results.append((name, "✅ PASSED"))
            chart_results.append((name, "passed", fig, spec))
        except Exception as e:
            failed += 1
            results.append((name, f"❌ FAILED: {e}"))
            chart_results.append((name, "failed", None, None))

    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    for name, status in results:
        print(f"   {name}: {status}")
    print(f"\n   {passed}/{passed + failed} tests passed")

    # --- Serve charts on localhost ---
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'test_output')
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, 'visualization_tests.html')

    with open(html_path, 'w') as f:
        f.write(build_html(chart_results))

    PORT = 8888
    os.chdir(output_dir)
    server = HTTPServer(('localhost', PORT), SimpleHTTPRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f'http://localhost:{PORT}/visualization_tests.html'
    print(f"\n🌐 Charts served at: {url}")
    print("   Press Ctrl+C to stop.")
    webbrowser.open(url)

    try:
        thread.join()
    except KeyboardInterrupt:
        print("\n👋 Server stopped.")
        server.shutdown()
