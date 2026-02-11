import json
import pandas as pd
from typing import TypedDict, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import vizro.plotly.express as vpx
import plotly.io as pio

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
LLM_MODEL = "llama3"
pio.templates.default = "vizro_dark"

# Supported chart types and their required/optional params
SUPPORTED_CHARTS = {
    "bar": {"required": ["x", "y"], "optional": ["color", "barmode"]},
    "line": {"required": ["x", "y"], "optional": ["color"]},
    "scatter": {"required": ["x", "y"], "optional": ["color", "size"]},
    "pie": {"required": ["names", "values"], "optional": ["color"]},
    "histogram": {"required": ["x"], "optional": ["color", "nbins"]},
    "box": {"required": ["y"], "optional": ["x", "color"]},
    "heatmap": {"required": ["x", "y", "z"], "optional": []},
    "treemap": {"required": ["path", "values"], "optional": ["color"]},
}

# ---------------------------------------------------------
# 1. CHART SPEC (Pydantic Validation)
# ---------------------------------------------------------
class ChartSpec(BaseModel):
    """The structured output the LLM must produce."""
    chart_type: str = Field(description="One of: bar, line, scatter, pie, histogram, box, heatmap, treemap")
    x: Optional[str] = Field(default=None, description="Column for x-axis")
    y: Optional[str] = Field(default=None, description="Column for y-axis")
    color: Optional[str] = Field(default=None, description="Column for color grouping")
    size: Optional[str] = Field(default=None, description="Column for size (scatter only)")
    names: Optional[str] = Field(default=None, description="Column for slice names (pie only)")
    values: Optional[str] = Field(default=None, description="Column for slice values (pie only)")
    z: Optional[str] = Field(default=None, description="Column for z-values (heatmap only)")
    path: Optional[list[str]] = Field(default=None, description="Hierarchy columns (treemap only)")
    nbins: Optional[int] = Field(default=None, description="Number of bins (histogram only)")
    barmode: Optional[str] = Field(default=None, description="'group' or 'stack' (bar only)")
    title: str = Field(default="Chart", description="Descriptive chart title")

# ---------------------------------------------------------
# 2. DEFINE THE STATE
# ---------------------------------------------------------
class VisualizationState(TypedDict):
    user_question: str          # Original user question
    df_columns: Optional[str]   # Column names + dtypes as text
    df_sample: Optional[str]    # First 5 rows as markdown
    df_row_count: Optional[int] # Total rows
    chart_spec_json: Optional[str]  # Raw JSON string from LLM
    chart_spec: Optional[dict]  # Validated ChartSpec as dict
    figure_json: Optional[str]  # Plotly figure as JSON
    error_message: Optional[str]
    retry_count: int

# ---------------------------------------------------------
# 3. DEFINE THE NODES
# ---------------------------------------------------------

def decide_chart_node(state: VisualizationState):
    """
    Uses Llama 3 to decide the best chart type and column mappings.
    """
    print("🎨 Deciding chart type...")

    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    template = """You are a chart-type selector. Pick the best chart for the question and data.

Question: {question}
Columns: {columns}
Sample: {sample}
Rows: {row_count}

Rules:
- bar: compare values across categories. x=category, y=value.
- line: trends over time. x=date/time, y=value.
- scatter: correlation between two numerics. x=num1, y=num2, color=optional category.
- pie: proportions/share/percentage with <8 categories. names=category, values=number.
- histogram: frequency distribution of ONE numeric column with many values. x=numeric_column. Only set x, leave y null.
- box: spread/outliers. y=numeric, x=grouping_category (REQUIRED if groups exist).
- heatmap: matrix. x=col1, y=col2, z=value.
- treemap: hierarchy. path=["parent","child"], values=number.

CRITICAL: Match chart to question intent, not just data shape.
- "proportion/share/breakdown" + few categories → pie
- "distribution/frequency" + many numeric values → histogram
- "spread/outliers/quartiles" + groups → box with x=group_column
- "correlation/relationship" → scatter with color=category if available

Reply with ONLY this JSON (no markdown, no explanation):
{{"chart_type":"...","x":null,"y":null,"color":null,"size":null,"names":null,"values":null,"z":null,"path":null,"nbins":null,"barmode":null,"title":"Descriptive Title"}}"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "question": state["user_question"],
        "columns": state["df_columns"],
        "sample": state["df_sample"],
        "row_count": state["df_row_count"],
    })

    # --- Parse and Validate ---
    try:
        # Clean up common LLM quirks
        clean = response.strip()
        clean = clean.replace("```json", "").replace("```", "").strip()

        # Find the JSON object in the response
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start == -1 or end == 0:
            # Try to recover truncated JSON by appending closing brace
            if start != -1 and end == 0:
                json_str = clean[start:] + "}"
            else:
                raise ValueError("No JSON object found in response")
        else:
            json_str = clean[start:end]

        # Fix trailing commas (common LLM quirk): ", }" or ",}" or ",\n}"
        import re as _re
        json_str = _re.sub(r',\s*}', '}', json_str)
        json_str = _re.sub(r',\s*]', ']', json_str)

        # Fix truncated JSON: if it's missing the closing brace, add it
        open_braces = json_str.count('{') - json_str.count('}')
        for _ in range(open_braces):
            json_str += '}'

        # Fix truncated key-value pairs: remove last incomplete entry if parsing fails
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            # Remove the last incomplete field and close the object
            last_comma = json_str.rfind(',')
            if last_comma != -1:
                json_str = json_str[:last_comma] + '}'
                json_str = _re.sub(r',\s*}', '}', json_str)
                parsed = json.loads(json_str)
            else:
                raise

        # Fix treemap quirk: LLM sometimes puts list in "x" instead of only "path"
        if parsed.get('chart_type') == 'treemap' and isinstance(parsed.get('x'), list):
            if not parsed.get('path'):
                parsed['path'] = parsed['x']
            parsed['x'] = None
            parsed['y'] = None

        # Fix treemap quirk: LLM sometimes puts hierarchy in "names" instead of "path"
        if parsed.get('chart_type') == 'treemap' and isinstance(parsed.get('names'), list) and len(parsed.get('names', [])) > 1:
            if not parsed.get('path'):
                parsed['path'] = parsed['names']
            parsed['names'] = None

        # Fix treemap/pie quirk: LLM wraps single column name in a list
        # e.g. "values": ["GDP_Billion"] → "values": "GDP_Billion"
        for field in ('names', 'values', 'x', 'y', 'z', 'color', 'size'):
            val = parsed.get(field)
            if isinstance(val, list) and len(val) == 1 and isinstance(val[0], str):
                parsed[field] = val[0]

        # Fix pie quirk: LLM puts literal data values instead of column name
        # e.g. "values": [42, 28, 22, 8] → detect this and find the right column
        if parsed.get('chart_type') == 'pie':
            val = parsed.get('values')
            if isinstance(val, list) and len(val) > 0 and not isinstance(val[0], str):
                # LLM put actual data instead of column name — find the numeric column
                col_names = [c.strip() for c in state["df_columns"].split('\n') if c.strip()]
                for col_line in col_names:
                    parts = col_line.lstrip('- ').split(':')
                    if len(parts) == 2:
                        col_name = parts[0].strip()
                        col_type = parts[1].strip()
                        if 'int' in col_type or 'float' in col_type:
                            parsed['values'] = col_name
                            break

        # Validate with Pydantic
        spec = ChartSpec.model_validate(parsed)

        # Check chart_type is supported
        if spec.chart_type not in SUPPORTED_CHARTS:
            raise ValueError(f"Unsupported chart type: {spec.chart_type}")

        # Auto-generate title if LLM returned the default
        if spec.title in ("Chart", "", None):
            # Create a title from the user question
            q = state["user_question"]
            spec.title = q[:80].title() if q else "Chart"

        print(f"   ✅ Chart spec: {spec.chart_type} (x={spec.x}, y={spec.y})")
        return {
            "chart_spec_json": json_str,
            "chart_spec": spec.model_dump(),
            "error_message": None,
        }

    except Exception as e:
        error_msg = f"Failed to parse chart spec: {e}\nLLM response: {response[:200]}"
        print(f"   ❌ {error_msg}")
        return {
            "error_message": error_msg,
            "retry_count": state["retry_count"] + 1,
        }


# ---------------------------------------------------------
# 4. ROUTER LOGIC
# ---------------------------------------------------------
def after_decide(state: VisualizationState):
    if state.get("chart_spec") is not None:
        return "render"
    if state["retry_count"] >= 3:
        print("   🛑 Max retries reached. Giving up.")
        return "give_up"
    return "retry_decide"


def after_render(state: VisualizationState):
    if state.get("figure_json") is not None:
        return "success"
    if state["retry_count"] >= 3:
        print("   🛑 Max retries reached. Giving up.")
        return "give_up"
    return "retry_decide"


# ---------------------------------------------------------
# 5. CHART RENDERER (Maps spec → vizro.plotly.express call)
# ---------------------------------------------------------
def build_figure(df: pd.DataFrame, spec: ChartSpec):
    """
    Deterministic: maps a ChartSpec to a vizro.plotly.express function call.
    Returns a Plotly Figure.
    """
    chart_fn_map = {
        "bar": vpx.bar,
        "line": vpx.line,
        "scatter": vpx.scatter,
        "pie": vpx.pie,
        "histogram": vpx.histogram,
        "box": vpx.box,
        "heatmap": vpx.density_heatmap,
        "treemap": vpx.treemap,
    }

    fn = chart_fn_map[spec.chart_type]

    # Build kwargs dynamically — only pass non-None values
    kwargs = {"data_frame": df, "title": spec.title}

    param_map = {
        "x": spec.x,
        "y": spec.y,
        "color": spec.color,
        "size": spec.size,
        "names": spec.names,
        "values": spec.values,
        "z": spec.z,
        "path": spec.path,
        "nbins": spec.nbins,
        "barmode": spec.barmode,
    }

    for param, value in param_map.items():
        if value is not None:
            kwargs[param] = value

    fig = fn(**kwargs)
    return fig


# ---------------------------------------------------------
# 6. VISUALIZATION AGENT (Wrapper Class)
# ---------------------------------------------------------
class VisualizationAgent:
    def __init__(self):
        pass  # Graph is built per-run (needs DataFrame closure)

    def run(self, df: pd.DataFrame, user_question: str):
        """
        Takes a DataFrame and user question, returns a Plotly figure.
        Returns: (success: bool, fig: Figure|None, spec: dict|None)
        """
        # --- Pre-populate data analysis ---
        col_info = "\n".join([
            f"  - {col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)
        ])
        sample_md = df.head(5).to_markdown()
        row_count = len(df)

        # --- Build render node with DataFrame closure ---
        def render_chart_with_data(state: VisualizationState):
            spec_dict = state["chart_spec"]
            spec = ChartSpec.model_validate(spec_dict)

            print(f"📊 Rendering {spec.chart_type} chart...")
            try:
                fig = build_figure(df, spec)
                print("   ✅ Chart rendered successfully!")
                return {
                    "figure_json": fig.to_json(),
                    "error_message": None,
                }
            except Exception as e:
                error_msg = f"Render error: {e}"
                print(f"   ❌ {error_msg}")
                return {
                    "error_message": error_msg,
                    "chart_spec": None,  # Force re-decide
                    "retry_count": state["retry_count"] + 1,
                }

        # --- Build the graph ---
        workflow = StateGraph(VisualizationState)

        workflow.add_node("decide", decide_chart_node)
        workflow.add_node("render", render_chart_with_data)

        workflow.set_entry_point("decide")

        workflow.add_conditional_edges("decide", after_decide, {
            "render": "render",
            "retry_decide": "decide",
            "give_up": END,
        })

        workflow.add_conditional_edges("render", after_render, {
            "success": END,
            "retry_decide": "decide",
            "give_up": END,
        })

        app = workflow.compile()

        # --- Run ---
        initial_state = {
            "user_question": user_question,
            "df_columns": col_info,
            "df_sample": sample_md,
            "df_row_count": row_count,
            "chart_spec_json": None,
            "chart_spec": None,
            "figure_json": None,
            "error_message": None,
            "retry_count": 0,
        }

        final_state = app.invoke(initial_state)

        if final_state.get("figure_json"):
            fig = pio.from_json(final_state["figure_json"])
            return True, fig, final_state["chart_spec"]
        else:
            return False, None, None
