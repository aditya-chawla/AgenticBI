"""
AgenticBI Vizro frontend.
Layout: chart dashboard (left ~62%) | chat (right ~38%), Cursor-style dark UI.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from datetime import datetime
from typing import Literal

import plotly.graph_objects as go
import plotly.io as pio
from dash import callback, dcc, html, Input, Output, State, no_update, ALL, MATCH, callback_context, clientside_callback
import vizro.models as vm
from vizro import Vizro

# ---------------------------------------------------------
# sys.path — _AGENTS_DIR first so agents' bare "from config import" works
# ---------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC         = os.path.join(_PROJECT_ROOT, "src")
_AGENTS_DIR  = os.path.join(_SRC, "agents")
_ASSETS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

for _p in (_AGENTS_DIR, _SRC, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pio.templates["agenticbi_dark"] = go.layout.Template(
    layout={
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#171a1f",
        "font": {"family": "Inter, Segoe UI, system-ui, sans-serif", "color": "#d6dde6", "size": 12},
        "colorway": ["#4f8cff", "#4fd1a5", "#f2b84b", "#ff7d73", "#9aa6ff", "#5fd3f3", "#c6d04f"],
        "margin": {"l": 56, "r": 22, "t": 48, "b": 54},
        "hoverlabel": {
            "bgcolor": "#232830",
            "bordercolor": "#3e4652",
            "font": {"color": "#f5f7fa"},
        },
        "xaxis": {
            "gridcolor": "rgba(148,163,184,0.11)",
            "linecolor": "rgba(148,163,184,0.18)",
            "zerolinecolor": "rgba(148,163,184,0.18)",
            "tickfont": {"color": "#9aa7b5"},
            "title": {"font": {"color": "#b8c2ce"}},
        },
        "yaxis": {
            "gridcolor": "rgba(148,163,184,0.13)",
            "linecolor": "rgba(148,163,184,0.18)",
            "zerolinecolor": "rgba(148,163,184,0.18)",
            "tickfont": {"color": "#9aa7b5"},
            "title": {"font": {"color": "#b8c2ce"}},
        },
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": "#aab5c2"},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    }
)
pio.templates.default = "agenticbi_dark"

# ---------------------------------------------------------
# Singletons
# ---------------------------------------------------------
_check_prompt_fn   = None
_orchestrator_inst = None


def _guardrails():
    global _check_prompt_fn
    if _check_prompt_fn is None:
        from guardrails_agent import check_prompt
        _check_prompt_fn = check_prompt
    return _check_prompt_fn


def _orchestrator():
    global _orchestrator_inst
    if _orchestrator_inst is None:
        from orchestrator_agent import OrchestratorAgent
        _orchestrator_inst = OrchestratorAgent()
    return _orchestrator_inst


# ---------------------------------------------------------
# Live progress: chips and Run-button narration update in real time during
# a pipeline run.
#
# The orchestrator pipeline (NL2SQL → SQL exec → viz → insights) takes
# 10–30 s. Rather than spawning a worker process (which crashes on macOS
# under `multiprocess` + HuggingFace fork-unsafety), we run the pipeline
# synchronously in the Flask thread that handles the Run callback, and
# update a thread-safe shared dict as each LangGraph node completes.
#
# A second callback driven by `dcc.Interval` polls that dict every ~700ms
# and pushes the latest chip state and button text to the browser. Flask
# `threaded=True` (the default in modern Flask/Werkzeug) lets the polling
# request run in a separate thread while the Run callback is still in
# progress, so the UI stays responsive.
# ---------------------------------------------------------
_NODE_TO_STAGE = {
    "generate_sql":      "sql",
    "execute_sql":       "query",
    "check_execution":   "query",
    "visualize":         "chart",
    "generate_insights": "memo",
}

_STAGE_BUTTON_TEXT = {
    None:    "Generating…",
    "sql":   "Running query…",
    "query": "Drawing chart…",
    "chart": "Writing memo…",
    "memo":  "Almost done…",
}

_PROGRESS_LOCK = threading.Lock()
_PROGRESS_STATE: dict = {
    # in_flight: True only while a run is being processed
    "in_flight": False,
    # tick: monotonically increasing — clients can detect "no change"
    "tick": 0,
    # the meta dict that _render_progress_chips consumes
    "meta": None,
    # current Run-button label
    "btn_text": "Run",
}


def _set_progress(meta: dict | None, btn_text: str, in_flight: bool):
    with _PROGRESS_LOCK:
        _PROGRESS_STATE["in_flight"] = in_flight
        _PROGRESS_STATE["meta"] = meta
        _PROGRESS_STATE["btn_text"] = btn_text
        _PROGRESS_STATE["tick"] += 1


def _get_progress_snapshot() -> dict:
    with _PROGRESS_LOCK:
        return dict(_PROGRESS_STATE)


def _streaming_meta(stages_done: set, running: str | None = None,
                    failed: bool = False, kind: str | None = None) -> dict:
    return {
        "sql":   "sql"   in stages_done,
        "query": "query" in stages_done,
        "chart": "chart" in stages_done,
        "memo":  "memo"  in stages_done,
        "running": running,
        "failed": failed,
        "kind":   kind,
    }


def _next_running(stages_done: set) -> str | None:
    for s in ("sql", "query", "chart", "memo"):
        if s not in stages_done:
            return s
    return None


# ---------------------------------------------------------
# Style tokens
# ---------------------------------------------------------
BG_MAIN   = "#0f1115"
BG_PANEL  = "#15181d"
BG_CARD   = "#1b2027"
BG_CARD_2 = "#141820"
BG_INPUT  = "#202631"
BG_MSG_US = "#1f3354"
BG_MSG_AI = "#1a1f26"
BORDER    = "1px solid #2a313a"
BORDER_SOFT = "1px solid #232a33"
ACCENT    = "#4f8cff"
ACCENT_SOFT = "#14233a"
TEXT      = "#e5e7eb"
TEXT_DIM  = "#8e99a8"
RADIUS    = "8px"
FONT      = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"

# ---------------------------------------------------------
# Suggested queries for empty state
# ---------------------------------------------------------
SUGGESTED_QUERIES = [
    "Show total sales by region",
    "Top 10 products by revenue",
    "Sales trend over time",
    "Employee vacation hours",
    "Online vs in-store orders",
]


_WORKSPACE_INFO_CACHE: dict | None = None


def _workspace_info() -> dict:
    """Cheap one-shot probe of the connected DB so the empty state and
    workspace badge can show real signals instead of decorative copy.
    Cached for the lifetime of the process — schemas don't change mid-session."""
    global _WORKSPACE_INFO_CACHE
    if _WORKSPACE_INFO_CACHE is not None:
        return _WORKSPACE_INFO_CACHE

    info = {"connected": False, "db": None, "tables": None, "schemas": None, "error": None}
    try:
        import psycopg2
        from config import DB_CONFIG
        conn = psycopg2.connect(connect_timeout=2, **DB_CONFIG)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT current_database()")
                info["db"] = cur.fetchone()[0]
                cur.execute(
                    "SELECT count(*) FROM information_schema.tables "
                    "WHERE table_schema NOT IN ('pg_catalog', 'information_schema')"
                )
                info["tables"] = cur.fetchone()[0]
                cur.execute(
                    "SELECT count(distinct table_schema) FROM information_schema.tables "
                    "WHERE table_schema NOT IN ('pg_catalog', 'information_schema')"
                )
                info["schemas"] = cur.fetchone()[0]
                info["connected"] = True
        finally:
            conn.close()
    except Exception as exc:
        info["error"] = str(exc)[:120]

    _WORKSPACE_INFO_CACHE = info
    return info


def _created_label() -> str:
    return datetime.now().strftime("%b %d, %I:%M %p").replace(" 0", " ")


def _records_from_json(df_json: str | None) -> list[dict]:
    if not df_json:
        return []
    try:
        data = json.loads(df_json)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _data_preview(records: list[dict], max_rows: int = 5):
    if not records:
        return html.Div("No row preview available.", className="detail-empty")

    columns = list(records[0].keys())[:5]
    return html.Table(
        [
            html.Thead(html.Tr([html.Th(col) for col in columns])),
            html.Tbody(
                [
                    html.Tr([html.Td(str(row.get(col, ""))) for col in columns])
                    for row in records[:max_rows]
                ]
            ),
        ],
        className="data-preview-table",
    )


_PROGRESS_STAGES = [("sql", "SQL"), ("query", "Query"), ("chart", "Chart"), ("memo", "Memo")]


def _render_progress_chips(meta: dict | None):
    """Stage chips reflect the run state. Recognised meta keys:

    - sql/query/chart/memo : booleans, whether that stage *completed*
    - running              : optional stage key (e.g. "query") currently in flight
    - failed               : whether the run failed overall
    - kind                 : failure flavour (used to amber-tint the SQL chip on
                             schema misses or full SQL failures)
    """
    if not meta:
        return [
            html.Span(label, className="step-chip")
            for _, label in _PROGRESS_STAGES
        ]

    failed = bool(meta.get("failed"))
    running = meta.get("running")
    chips = []
    for key, label in _PROGRESS_STAGES:
        classes = ["step-chip"]
        if meta.get(key):
            classes.append("completed")
        elif key == running:
            classes.append("running")
        elif failed and key == "sql":
            classes.append("failed")
        chips.append(html.Span(label, className=" ".join(classes)))
    return chips


def _polish_figure(fig, title: str, featured: bool = False):
    # The chart card already shows the title in its own header; suppressing the
    # in-canvas Plotly title removes duplication and frees vertical space.
    fig.update_layout(
        template="agenticbi_dark",
        title=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#171a1f",
        margin={"l": 62 if featured else 52, "r": 24, "t": 24 if featured else 18, "b": 58},
        font={"family": "Inter, Segoe UI, system-ui, sans-serif", "color": "#d6dde6"},
    )

    # Re-apply currency tick formatting after JSON deserialisation. The viz
    # agent sets tickprefix="$" but tickformat="~s" gets dropped through the
    # Plotly JSON round-trip, so we redo it here with values that survive.
    for axis_attr in ("xaxis", "yaxis"):
        axis = getattr(fig.layout, axis_attr, None)
        if axis is not None and getattr(axis, "tickprefix", "") == "$":
            axis.tickformat = "~s"
    fig.update_xaxes(
        automargin=True,
        gridcolor="rgba(148,163,184,0.10)",
        linecolor="rgba(148,163,184,0.20)",
        tickfont={"color": "#9aa7b5"},
    )
    fig.update_yaxes(
        automargin=True,
        gridcolor="rgba(148,163,184,0.13)",
        linecolor="rgba(148,163,184,0.20)",
        tickfont={"color": "#9aa7b5"},
    )
    for trace_index, trace in enumerate(fig.data):
        if getattr(trace, "type", "") == "bar":
            trace.update(
                marker={
                    "line": {"width": 0},
                    "opacity": 0.9,
                },
                hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>",
            )
        elif getattr(trace, "type", "") in {"pie", "funnelarea"}:
            trace.update(
                textfont={"color": "#f5f7fa"},
                hovertemplate="<b>%{label}</b><br>%{value}<extra></extra>",
            )
        elif getattr(trace, "type", "") == "scatter":
            trace.update(
                line={"width": 3},
                marker={"size": 8, "line": {"width": 0}},
                hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>",
            )
    return fig


# ---------------------------------------------------------
# Custom Vizro component
# ---------------------------------------------------------
class AgenticBIPage(vm.VizroBaseModel):
    type: Literal["agentic_bi_page"] = "agentic_bi_page"

    def build(self):
        S = self.id  # id prefix

        return html.Div(
            [
                # ── Hidden state store (local persistence) ───────
                dcc.Store(
                    id=f"{S}_store",
                    data={"charts": [], "messages": [], "all_charts": []},
                    storage_type="local",
                ),

                # ── Live-progress poll: ticks while a pipeline is running so
                #    chips and Run-button text update mid-stream. Dormant
                #    when no run is in flight.
                dcc.Interval(
                    id=f"{S}_progress_tick",
                    interval=600,    # ms
                    disabled=False,
                    n_intervals=0,
                ),

                # ── Hidden tick counter — bumped each time the polling
                #    callback sees a real change. Used as a tracking signal.
                dcc.Store(id=f"{S}_progress_tick_seen", data=0),

                # ── Run-completed pulse — bumped by poll_progress when it
                #    detects the in-flight flag flip from True → False, so
                #    render_ui can refresh to pick up the new store data.
                dcc.Store(id=f"{S}_run_completed_tick", data=0),


                # ── Loading overlay ─────────────────────────────
                dcc.Loading(
                    id=f"{S}_loading",
                    type="circle",
                    color=ACCENT,
                    children=html.Div(
                        id=f"{S}_loading_target",
                        style={
                            "minHeight": "1px",
                            "position": "absolute",
                            "top": "0",
                            "left": "0",
                        },
                    ),
                    style={
                        "position": "absolute",
                        "top": "50%",
                        "left": "50%",
                        "transform": "translate(-50%, -50%)",
                        "zIndex": "20",
                    },
                ),

                # ── Top command bar ────────────────────────────
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("AgenticBI", className="app-title"),
                                html.Span(
                                    id=f"{S}_workspace_badge",
                                    children="Connected",
                                    className="status-pill ok",
                                ),
                            ],
                            className="brand-block",
                        ),
                        html.Div(
                            [
                                dcc.Input(
                                    id=f"{S}_chat_input",
                                    type="text",
                                    placeholder="Ask anything about your data — try \"Top 10 products by revenue\"",
                                    debounce=False,
                                    n_submit=0,
                                    disabled=False,
                                    className="command-input",
                                ),
                                html.Kbd("/", className="kbd-hint"),
                                html.Button(
                                    "Run",
                                    id=f"{S}_send_btn",
                                    n_clicks=0,
                                    disabled=False,
                                    className="command-button",
                                ),
                            ],
                            className="command-box",
                        ),
                    ],
                    id=f"{S}_command_bar",
                    className="command-bar",
                ),

                # ── Main row: LEFT (charts) | RIGHT (insights) ──
                html.Div(
                    [
                        # ═══ LEFT PANEL — Dashboard ═══════════════
                        html.Div(
                            [
                                # Left panel header: label + filter + sort
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Span("DASHBOARD", className="panel-eyebrow"),
                                                html.Div(
                                                    id=f"{S}_progress_chips",
                                                    children=_render_progress_chips(None),
                                                    className="progress-steps",
                                                ),
                                            ],
                                            className="panel-title-block",
                                        ),
                                        dcc.Input(
                                            id=f"{S}_filter_query",
                                            type="text",
                                            placeholder="Filter charts by title or query",
                                            debounce=True,
                                            className="filter-input",
                                        ),
                                        dcc.Dropdown(
                                            id=f"{S}_sort_by",
                                            options=[
                                                {"label": "Latest first", "value": "latest"},
                                                {"label": "A → Z",        "value": "az"},
                                                {"label": "Z → A",        "value": "za"},
                                            ],
                                            value="latest",
                                            clearable=False,
                                            className="sort-dropdown",
                                            style={"width": "150px", "fontSize": "12px"},
                                        ),
                                    ],
                                    className="panel-header",
                                ),
                                # Charts scroll area
                                html.Div(
                                    id=f"{S}_dashboard",
                                    style={
                                        "flex": "1",
                                        "overflowY": "auto",
                                        "padding": "16px",
                                    },
                                ),
                            ],
                            style={
                                "flex": "1",
                                "display": "flex",
                                "flexDirection": "column",
                                "background": BG_MAIN,
                                "minWidth": "0",
                                "height": "100%",
                            },
                        ),

                        # ── Vertical divider ─────────────────────
                        html.Div(
                            style={
                                "width": "1px",
                                "background": "#30353b",
                                "flexShrink": "0",
                            }
                        ),

                        # ═══ RIGHT PANEL — Insight timeline ═══════
                        html.Div(
                            [
                                # Chat header with clear button
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Span("INSIGHTS", className="panel-eyebrow"),
                                                html.Span("Question-to-answer trail", className="panel-subtitle"),
                                            ],
                                            className="panel-title-block",
                                        ),
                                        html.Button(
                                            "Clear",
                                            id=f"{S}_clear_btn",
                                            n_clicks=0,
                                            className="subtle-button",
                                        ),
                                    ],
                                    className="panel-header insight-header",
                                ),
                                # Messages scroll area
                                html.Div(
                                    id=f"{S}_chat_messages",
                                    style={
                                        "flex": "1",
                                        "overflowY": "auto",
                                        "padding": "14px",
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "gap": "10px",
                                        "fontFamily": FONT,
                                    },
                                ),
                            ],
                            className="chat-panel",
                            style={
                                "width": "36%",
                                "minWidth": "300px",
                                "maxWidth": "560px",
                                "flexShrink": "0",
                                "display": "flex",
                                "flexDirection": "column",
                                "background": BG_PANEL,
                                "height": "100%",
                            },
                        ),
                    ],
                    className="main-layout",
                    style={
                        "display": "flex",
                        "flex": "1",
                        "height": "100%",
                        "minHeight": "0",
                        "overflow": "hidden",
                    },
                ),
            ],
            id=self.id,
            style={
                "display": "flex",
                "flexDirection": "column",
                "height": "100vh",
                "width": "100vw",
                "background": BG_MAIN,
                "color": TEXT,
                "fontFamily": FONT,
                "overflow": "hidden",
                "position": "fixed",
                "top": "0",
                "left": "0",
            },
        )


vm.Page.add_type("components", AgenticBIPage)


# ---------------------------------------------------------
# Callback 1 — Send / Enter → pipeline → update Store
#
# Runs synchronously in the Flask request thread. While streaming through the
# LangGraph pipeline, it pushes per-stage updates to _PROGRESS_STATE which a
# separate dcc.Interval-driven callback polls and reflects in the chips +
# Run button. Avoids the macOS multiprocess fork issues entirely.
# ---------------------------------------------------------
@callback(
    Output("agentic_bi_page_store", "data"),
    Output("agentic_bi_page_chat_input", "value"),
    Output("agentic_bi_page_chat_input", "disabled"),
    Output("agentic_bi_page_send_btn", "disabled"),
    Output("agentic_bi_page_send_btn", "children"),
    Output("agentic_bi_page_loading_target", "children"),
    Input("agentic_bi_page_send_btn", "n_clicks"),
    Input("agentic_bi_page_chat_input", "n_submit"),
    Input({"type": "suggested-query", "index": ALL}, "n_clicks"),
    Input({"type": "suggested-query-chat", "index": ALL}, "n_clicks"),
    State("agentic_bi_page_chat_input", "value"),
    State("agentic_bi_page_store", "data"),
    prevent_initial_call=True,
)
def on_send(_clicks, _submit, _suggested_dashboard, _suggested_chat, user_text, store):
    ctx = callback_context
    
    # Abort if the callback was fired by a component simply being added to the DOM (value is None)
    if not ctx.triggered or ctx.triggered[0].get("value") is None:
        return no_update, no_update, no_update, no_update, no_update, no_update

    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    question = ""

    if triggered.startswith("{"):
        try:
            trigger_id = json.loads(triggered)
            if trigger_id.get("type") in {"suggested-query", "suggested-query-chat"}:
                question = SUGGESTED_QUERIES[int(trigger_id["index"])]
        except Exception:
            question = ""
    else:
        question = (user_text or "").strip()

    if not question:
        return no_update, no_update, no_update, no_update, no_update, no_update

    store    = store or {"charts": [], "messages": [], "all_charts": []}
    charts   = list(store.get("charts", []))
    all_charts = list(store.get("all_charts", []))
    messages = list(store.get("messages", []))

    messages.append({"role": "user", "content": question, "timestamp": uuid.uuid4().hex[:8]})

    # Guardrails
    try:
        allowed, deny_msg = _guardrails()(question)
    except Exception:
        allowed, deny_msg = True, ""

    if not allowed:
        messages.append({
            "role": "assistant",
            "content": deny_msg,
            "meta": {"sql": False, "query": False, "chart": False, "memo": True, "failed": True, "kind": "blocked"},
        })
        return {**store, "charts": charts, "all_charts": all_charts, "messages": messages}, "", False, False, "Run", ""

    # Build conversation history for context
    history = [m["content"] for m in messages if m.get("role") == "user"]

    # --- Stream the orchestrator and push chip updates as stages complete ---
    stages_done: set = set()

    def _push_progress():
        running = _next_running(stages_done)
        meta = _streaming_meta(stages_done, running=running)
        _set_progress(meta, _STAGE_BUTTON_TEXT.get(running, "Generating…"), in_flight=True)

    # Initial state — SQL stage about to start
    _set_progress(_streaming_meta(set(), running="sql"), _STAGE_BUTTON_TEXT["sql"], in_flight=True)

    result: dict | None = None
    try:
        gen = _orchestrator().run_streaming(question, conversation_history=history)
        try:
            while True:
                node_name, snapshot = next(gen)
                stage = _NODE_TO_STAGE.get(node_name)

                # Per-stage success rules: only mark a stage complete when the
                # state carries the artifact that stage actually produced
                # (except memo — the generate_insights node always runs last,
                # so we mark it done whenever it fires).
                if stage == "sql" and snapshot.get("sql_query"):
                    stages_done.add("sql")
                elif stage == "query" and snapshot.get("sql_success") and snapshot.get("result_dict"):
                    stages_done.add("sql")
                    stages_done.add("query")
                elif stage == "chart" and snapshot.get("viz_success"):
                    stages_done.update({"sql", "query", "chart"})
                elif stage == "memo":
                    stages_done.update({"sql", "query", "chart", "memo"})

                _push_progress()
        except StopIteration as stop:
            result = stop.value
    except Exception as exc:
        _set_progress(None, "Run", in_flight=False)
        messages.append({
            "role": "assistant",
            "content": f"Pipeline error: {exc}",
            "meta": {"sql": False, "query": False, "chart": False, "memo": True, "failed": True, "kind": "crash"},
        })
        return {**store, "charts": charts, "all_charts": all_charts, "messages": messages}, "", False, False, "Run", ""

    if result is None:
        result = {"success": False, "error": "Orchestrator stream produced no result"}

    # Push a final chip snapshot reflecting how the run actually ended, then
    # flip in_flight=False. The next poll picks this up and bumps the
    # run_completed_tick so render_ui re-fires.
    raw_err = (result.get("error") or "").lower() if not result.get("success") else ""
    final_kind = None
    if not result.get("success"):
        final_kind = "schema_miss" if "insufficient schema context" in raw_err else "sql_failed"
    final_meta = _streaming_meta(
        stages_done,
        running=None,
        failed=not result.get("success"),
        kind=final_kind,
    )
    _set_progress(final_meta, "Run", in_flight=False)

    # Per-stage flags for the final memo meta
    has_sql   = bool(result.get("sql"))
    has_data  = result.get("df") is not None and len(result["df"]) > 0
    has_chart = result.get("figure") is not None
    has_memo  = bool(result.get("markdown"))

    if result.get("success") and has_chart:
        title = (result.get("chart_spec") or {}).get("title") or "Chart"
        chart_id = str(uuid.uuid4())[:8]
        chart_data = {
            "id":          chart_id,
            "query":       question,
            "title":       title,
            "figure_json": result["figure"].to_json(),
            "sql":         result.get("sql"),
            "df_json":     result["df"].to_json(orient="records") if result.get("df") is not None else None,
            "row_count":   int(len(result["df"])) if result.get("df") is not None else 0,
            "chart_type":  (result.get("chart_spec") or {}).get("chart_type"),
            "created_at":  _created_label(),
            "timestamp":   uuid.uuid4().hex[:8],
        }
        charts.insert(0, chart_data)
        all_charts.insert(0, chart_data)
        reply = result.get("markdown") or f"Chart ready: **{title}**"
        meta = {"sql": True, "query": True, "chart": True, "memo": has_memo, "failed": False, "kind": "ok"}

    elif result.get("success") and result.get("viz_failed"):
        reply = (
            "Query ran but a chart couldn't be rendered for these results.\n\n"
            + (result.get("viz_error") or "")
            + ("\n\n" + result["markdown"] if result.get("markdown") else "")
        )
        meta = {"sql": True, "query": True, "chart": False, "memo": has_memo, "failed": True, "kind": "viz_failed"}

    else:
        raw = result.get("error") or "Could not answer that. Try rephrasing."
        if "insufficient schema context" in raw.lower():
            reply = (
                "I couldn't answer this from the connected dataset.\n\n"
                "The columns or relationships you're asking about don't appear "
                "in the AdventureWorks schema I have indexed. Try rephrasing "
                "with terms closer to the data — for example, sales territory "
                "instead of acquisition channel."
            )
            kind = "schema_miss"
        else:
            reply = "Couldn't run that query.\n\n" + raw
            kind = "sql_failed"
        meta = {"sql": has_sql, "query": False, "chart": False, "memo": False, "failed": True, "kind": kind}

    messages.append({"role": "assistant", "content": reply, "meta": meta})
    return {**store, "charts": charts, "all_charts": all_charts, "messages": messages}, "", False, False, "Run", ""




# ---------------------------------------------------------
# Callback 1b — Clear chat
# ---------------------------------------------------------
@callback(
    Output("agentic_bi_page_store", "data", allow_duplicate=True),
    Input("agentic_bi_page_clear_btn", "n_clicks"),
    State("agentic_bi_page_store", "data"),
    prevent_initial_call=True,
)
def on_clear(_clicks, store):
    store = store or {"charts": [], "messages": [], "all_charts": []}
    return {**store, "charts": [], "messages": [], "all_charts": []}


# ---------------------------------------------------------
# Callback 1c — Delete chart
# ---------------------------------------------------------
@callback(
    Output("agentic_bi_page_store", "data", allow_duplicate=True),
    Input({"type": "delete-chart", "index": ALL}, "n_clicks"),
    State("agentic_bi_page_store", "data"),
    prevent_initial_call=True,
)
def on_delete_chart(n_clicks, store):
    if not n_clicks or not any(n_clicks):
        return no_update
    store = store or {"charts": [], "messages": [], "all_charts": []}
    charts = list(store.get("charts", []))
    all_charts = list(store.get("all_charts", []))
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
    chart_id = json.loads(triggered)["index"]
    charts = [c for c in charts if c.get("id") != chart_id]
    all_charts = [c for c in all_charts if c.get("id") != chart_id]
    return {**store, "charts": charts, "all_charts": all_charts}


# ---------------------------------------------------------
# Callback 2 — Store / filter / sort → render UI
# ---------------------------------------------------------
@callback(
    Output("agentic_bi_page_dashboard", "children"),
    Output("agentic_bi_page_chat_messages", "children"),
    Output("agentic_bi_page_workspace_badge", "children"),
    Output("agentic_bi_page_workspace_badge", "className"),
    Input("agentic_bi_page_store", "data"),
    Input("agentic_bi_page_filter_query", "value"),
    Input("agentic_bi_page_sort_by", "value"),
    Input("agentic_bi_page_run_completed_tick", "data"),
)
def render_ui(store, filter_q, sort_by, _completed_tick):
    store    = store or {"charts": [], "messages": [], "all_charts": []}
    all_charts = list(store.get("all_charts", []))
    charts   = list(store.get("charts", []))
    messages = list(store.get("messages", []))

    # Track last assistant message's meta — used for the workspace badge
    # and the failed-memo styling. Chips are now owned solely by the polling
    # callback (live during runs, default afterward).
    last_assistant_meta = None
    for m in reversed(messages):
        if m.get("role") == "assistant":
            last_assistant_meta = m.get("meta")
            break

    # Filter
    fq = (filter_q or "").strip().lower()
    if fq:
        charts = [
            c for c in charts
            if fq in (c.get("query") or "").lower()
            or fq in (c.get("title") or "").lower()
        ]

    # Sort
    if sort_by == "az":
        charts = sorted(charts, key=lambda c: (c.get("title") or "").lower())
    elif sort_by == "za":
        charts = sorted(charts, key=lambda c: (c.get("title") or "").lower(), reverse=True)

    # Single live status row: chart count + last-run summary. Replaces the
    # previous 4-up KPI grid (which mixed counters with an event card).
    kpi_strip = None
    if all_charts:
        latest = all_charts[0]
        latest_meta = (last_assistant_meta or {})
        latest_failed = bool(latest_meta.get("failed"))
        kpi_strip = html.Div(
            [
                html.Span(f"{len(all_charts)} chart{'s' if len(all_charts) != 1 else ''}", className="status-counter"),
                html.Span("·", className="status-divider"),
                html.Span(
                    [
                        html.Span("Last run · ", className="status-key"),
                        html.Strong(latest.get("title") or "Ready", className="status-value"),
                        html.Span(
                            f"  ({latest.get('row_count', 0)} rows · {latest.get('created_at', 'just now')})",
                            className="status-meta",
                        ),
                    ],
                    className="status-last",
                ),
                html.Span(
                    "Most recent question failed" if latest_failed else "All recent runs OK",
                    className=f"status-health {'warn' if latest_failed else 'ok'}",
                ),
            ],
            className="status-strip",
        )

    # ── Dashboard ───────────────────────────────────────
    if not charts:
        if all_charts and fq:
            # Charts exist but filter hides them
            empty_msg = f'No charts match "{fq}". Try a different filter.'
            dash_out = [
                kpi_strip,
                html.Div(
                    [
                        html.Div("No matching charts", className="empty-title"),
                        html.Div(
                            empty_msg,
                            className="empty-copy",
                        ),
                    ],
                    className="empty-panel",
                )
            ]
        else:
            # Truly empty — show proof of life + suggested queries
            info = _workspace_info()
            if info["connected"]:
                conn_pills = [
                    html.Span([html.Span("●", className="dot ok"), info["db"] or "postgres"], className="conn-pill"),
                    html.Span(f"{info['tables']} tables", className="conn-pill muted"),
                    html.Span(f"{info['schemas']} schemas", className="conn-pill muted"),
                ]
            else:
                conn_pills = [
                    html.Span([html.Span("●", className="dot err"), "Database unreachable"], className="conn-pill error"),
                    html.Span(info.get("error") or "", className="conn-pill muted"),
                ]
            dash_out = [
                html.Div(
                    [
                        html.Div(conn_pills, className="conn-row"),
                        html.Div("Ask anything about your data.", className="empty-title"),
                        html.Div(
                            "AgenticBI generates SQL, runs the query, draws the chart, and writes a short narrative — all from one prompt.",
                            className="empty-copy",
                        ),
                        html.Div("Try one of these to see the pipeline in action:", className="empty-cta"),
                        html.Div(
                            [
                                html.Button(
                                    q,
                                    id={"type": "suggested-query", "index": i},
                                    n_clicks=0,
                                    className="suggested-chip",
                                )
                                for i, q in enumerate(SUGGESTED_QUERIES)
                            ],
                            className="suggestion-row",
                        ),
                    ],
                    className="empty-panel",
                )
            ]
    else:
        chart_cards = []
        for i, c in enumerate(charts):
            try:
                fig = pio.from_json(c["figure_json"])
                featured = i == 0
                title = c.get("title") or "Chart"
                fig = _polish_figure(fig, title, featured=featured)
                chart_id = c.get("id", "")
                records = _records_from_json(c.get("df_json"))
                row_count = c.get("row_count")
                if row_count is None:
                    row_count = len(records)
                chart_type = (c.get("chart_type") or "chart").replace("_", " ").title()
                chart_cards.append(
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(title, className="chart-title"),
                                            html.Div(
                                                [
                                                    html.Span(chart_type, className="chart-meta-label"),
                                                    html.Span(f"{row_count} rows", className="chart-meta-label neutral"),
                                                    html.Span(c.get("query") or "", className="chart-meta-query"),
                                                ],
                                                className="chart-meta",
                                            ),
                                        ],
                                        className="chart-title-block",
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "✕",
                                                id={"type": "delete-chart", "index": chart_id},
                                                n_clicks=0,
                                                title="Remove from dashboard",
                                                className="chart-action-button danger",
                                            ),
                                        ],
                                        className="chart-actions",
                                    ),
                                ],
                                className="chart-card-header",
                            ),
                            html.Div(
                                dcc.Graph(
                                    id={"type": "chart-graph", "index": chart_id},
                                    figure=fig,
                                    config={
                                        "displayModeBar": False,
                                        "responsive": True,
                                        "toImageButtonOptions": {
                                            "format": "png",
                                            "filename": f"chart_{chart_id}",
                                        },
                                    },
                                    style={
                                        "height": "390px" if featured else "250px",
                                        "minHeight": "300px" if featured else "220px",
                                    },
                                ),
                                className="chart-figure-wrap",
                            ),
                            html.Div(
                                [
                                    html.Details(
                                        [
                                            html.Summary("SQL"),
                                            html.Pre(c.get("sql") or "No SQL available."),
                                        ],
                                        className="detail-drawer",
                                    ),
                                    html.Details(
                                        [
                                            html.Summary("Data preview"),
                                            _data_preview(records),
                                        ],
                                        className="detail-drawer",
                                    ),
                                ],
                                className="chart-drawers",
                            ),
                        ],
                        className="chart-card featured-chart-card" if featured else "chart-card",
                    )
                )
            except Exception:
                pass
        dash_out = [kpi_strip, html.Div(chart_cards, className="chart-grid")]

    # ── Chat messages ────────────────────────────────────
    if not messages:
        chat_out = [
            html.Div(
                [
                    html.Div("How the trail works", className="memo-title"),
                    html.Div(
                        "Each question generates SQL, runs it, draws a chart, and writes a short narrative — they all land here. The latest sits on top; older runs collapse to one line and expand on click.",
                        className="memo-copy",
                    ),
                    html.Div(
                        [
                            html.Span("01 ", className="step-num"), "Question is parsed and grounded in the schema.",
                        ],
                        className="memo-step",
                    ),
                    html.Div(
                        [
                            html.Span("02 ", className="step-num"), "SQL is generated and executed against Postgres.",
                        ],
                        className="memo-step",
                    ),
                    html.Div(
                        [
                            html.Span("03 ", className="step-num"), "Chart is drawn from the result and a memo is written.",
                        ],
                        className="memo-step",
                    ),
                ],
                className="welcome-panel",
            )
        ]
    else:
        chat_out = []
        exchanges = []
        pending_question = None
        for m in messages:
            if m.get("role") == "user":
                pending_question = m.get("content", "")
            elif m.get("role") == "assistant":
                exchanges.append({
                    "question": pending_question,
                    "answer": m.get("content", ""),
                    "meta": m.get("meta") or {},
                })
                pending_question = None

        for idx, exchange in enumerate(reversed(exchanges)):
            is_latest = idx == 0
            meta = exchange.get("meta") or {}
            failed = bool(meta.get("failed"))

            card_classes = ["memo-card"]
            card_classes.append("primary-memo" if is_latest else "history-memo")
            if failed:
                card_classes.append("failed-memo")

            if failed:
                kind = meta.get("kind") or "failed"
                status_label = {
                    "schema_miss": "Out of scope",
                    "sql_failed":  "Query failed",
                    "viz_failed":  "Chart not rendered",
                    "blocked":     "Blocked",
                    "crash":       "Pipeline error",
                }.get(kind, "Failed")
                status_class = "memo-status failed"
            else:
                status_label = "Generated from live SQL"
                status_class = "memo-status"

            question_text = exchange.get("question") or "Question unavailable"
            answer_md = exchange.get("answer") or ""

            # Latest memo renders fully expanded; older memos collapse to a
            # one-line summary that opens on click.
            if is_latest:
                body = [
                    html.Div(
                        [
                            html.Span("Latest memo", className="message-label"),
                            html.Span(status_label, className=status_class),
                        ],
                        className="memo-card-topline",
                    ),
                    html.Div(question_text, className="memo-question"),
                    dcc.Markdown(answer_md, className="insight-markdown"),
                ]
                card = html.Div(body, className=" ".join(card_classes))
            else:
                preview = (question_text or "").strip()
                if len(preview) > 80:
                    preview = preview[:77] + "…"
                card = html.Details(
                    [
                        html.Summary(
                            [
                                html.Span(preview, className="history-question"),
                                html.Span(status_label, className=f"{status_class} small"),
                            ],
                            className="history-summary",
                        ),
                        dcc.Markdown(answer_md, className="insight-markdown muted"),
                    ],
                    className=" ".join(card_classes + ["collapsed-memo"]),
                )

            chat_out.append(card)

    # Workspace badge — reflects connection + most recent run state
    info = _workspace_info()
    if not info["connected"]:
        badge_text = "DB unreachable"
        badge_class = "status-pill err"
    elif last_assistant_meta and last_assistant_meta.get("failed"):
        badge_text = "Last run failed"
        badge_class = "status-pill warn"
    else:
        badge_text = "Connected"
        badge_class = "status-pill ok"

    return dash_out, chat_out, badge_text, badge_class


# ---------------------------------------------------------
# Live-progress polling — ticks while a run is in flight, pushes chip + button
# updates from the shared in-process state dict.
# ---------------------------------------------------------
@callback(
    Output("agentic_bi_page_progress_chips", "children"),
    Output("agentic_bi_page_send_btn", "children", allow_duplicate=True),
    Output("agentic_bi_page_progress_tick_seen", "data"),
    Output("agentic_bi_page_run_completed_tick", "data"),
    Input("agentic_bi_page_progress_tick", "n_intervals"),
    State("agentic_bi_page_progress_tick_seen", "data"),
    State("agentic_bi_page_run_completed_tick", "data"),
    prevent_initial_call=True,
)
def poll_progress(_n, last_seen, completed_tick):
    snap = _get_progress_snapshot()
    if snap["tick"] == (last_seen or 0):
        return no_update, no_update, no_update, no_update

    # Run just finished — write the final chip state and bump the completion
    # tick so render_ui re-reads the store and refreshes the dashboard.
    if not snap["in_flight"]:
        chips = _render_progress_chips(snap["meta"]) if snap["meta"] else no_update
        return chips, "Run", snap["tick"], (completed_tick or 0) + 1

    return _render_progress_chips(snap["meta"]), snap["btn_text"], snap["tick"], no_update


# ---------------------------------------------------------
# Clientside callbacks — instant feedback that doesn't need server roundtrip
# ---------------------------------------------------------

# 1) On Run click: instantly disable input/button and narrate the button.
#    Adds .is-running to the chips container only briefly — once the worker
#    starts streaming, set_progress overwrites the chip children with real
#    per-stage state and the per-chip .running animation takes over.
clientside_callback(
    """
    function(n_clicks, n_submit) {
        if (!n_clicks && !n_submit) {
            return [window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update];
        }
        const chips = document.getElementById('agentic_bi_page_progress_chips');
        if (chips) chips.classList.add('is-running');
        return [true, true, 'Generating…'];
    }
    """,
    [
        Output("agentic_bi_page_chat_input", "disabled", allow_duplicate=True),
        Output("agentic_bi_page_send_btn", "disabled", allow_duplicate=True),
        Output("agentic_bi_page_send_btn", "children", allow_duplicate=True),
    ],
    [
        Input("agentic_bi_page_send_btn", "n_clicks"),
        Input("agentic_bi_page_chat_input", "n_submit"),
    ],
    prevent_initial_call=True,
)

# 2) When chips children update (set_progress or final render), remove the
#    is-running spin-up class so it doesn't compete with real .running chips.
clientside_callback(
    """
    function(_children) {
        const chips = document.getElementById('agentic_bi_page_progress_chips');
        if (chips) chips.classList.remove('is-running');
        return '';
    }
    """,
    Output("agentic_bi_page_progress_chips", "title"),
    Input("agentic_bi_page_progress_chips", "children"),
    prevent_initial_call=True,
)

# 3) Keyboard shortcut: pressing "/" anywhere outside an input focuses the
#    question box. Registered once via a module-scoped guard.
clientside_callback(
    """
    function(_data) {
        if (window.__agenticbi_kb_bound) return '';
        window.__agenticbi_kb_bound = true;
        document.addEventListener('keydown', function(e) {
            if (e.key !== '/') return;
            const tag = (document.activeElement && document.activeElement.tagName) || '';
            if (tag === 'INPUT' || tag === 'TEXTAREA') return;
            const input = document.getElementById('agentic_bi_page_chat_input');
            if (input) {
                e.preventDefault();
                input.focus();
                input.select && input.select();
            }
        });
        return '';
    }
    """,
    Output("agentic_bi_page_chat_input", "title"),
    Input("agentic_bi_page_store", "data"),
)




# ---------------------------------------------------------
# Build and run
# ---------------------------------------------------------
def run_app(host: str = "0.0.0.0", port: int = 8050, debug: bool = False) -> None:
    page = vm.Page(
        title="",          # suppress Vizro page title heading
        layout=vm.Flex(),
        components=[AgenticBIPage(id="agentic_bi_page")],
    )
    dashboard = vm.Dashboard(
        pages=[page],
        title="",          # suppress Vizro dashboard header title
    )
    app = Vizro(assets_folder=_ASSETS_DIR).build(dashboard)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_app(debug=os.environ.get("DEBUG", "0") == "1")
