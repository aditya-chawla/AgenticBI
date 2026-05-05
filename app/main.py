"""
AgenticBI Vizro frontend.
Layout: chart dashboard (left ~62%) | chat (right ~38%), Cursor-style dark UI.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Literal

import plotly.io as pio
from dash import callback, dcc, html, Input, Output, State, no_update, ALL, callback_context
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

pio.templates.default = "vizro_dark"

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
# Style tokens (Cursor / VS Code dark palette)
# ---------------------------------------------------------
BG_MAIN   = "#1e1e1e"
BG_PANEL  = "#252526"
BG_INPUT  = "#3c3c3c"
BG_MSG_US = "#2d4a7a"
BG_MSG_AI = "#2d2d2d"
BORDER    = "1px solid #3c3c3c"
ACCENT    = "#0e639c"
TEXT      = "#cccccc"
TEXT_DIM  = "#6a6a6a"
RADIUS    = "6px"
FONT      = "'Segoe UI', system-ui, -apple-system, sans-serif"

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

                # ── Loading overlay ─────────────────────────────
                dcc.Loading(
                    id=f"{S}_loading",
                    type="circle",
                    color=ACCENT,
                    children=html.Div(id=f"{S}_loading_target", style={"display": "none"}),
                    style={"position": "absolute", "top": "50%", "left": "50%", "transform": "translate(-50%, -50%)"},
                ),

                # ── Main row: LEFT (charts) | RIGHT (chat) ─────
                html.Div(
                    [
                        # ═══ LEFT PANEL — Dashboard ═══════════════
                        html.Div(
                            [
                                # Left panel header: label + filter + sort
                                html.Div(
                                    [
                                        html.Span(
                                            "CHARTS",
                                            style={
                                                "fontSize": "11px",
                                                "color": TEXT_DIM,
                                                "letterSpacing": "1.5px",
                                                "fontWeight": "600",
                                                "marginRight": "auto",
                                            },
                                        ),
                                        dcc.Input(
                                            id=f"{S}_filter_query",
                                            type="text",
                                            placeholder="Filter charts…",
                                            debounce=True,
                                            style={
                                                "width": "150px",
                                                "padding": "4px 8px",
                                                "fontSize": "12px",
                                                "background": BG_INPUT,
                                                "color": TEXT,
                                                "border": BORDER,
                                                "borderRadius": RADIUS,
                                                "outline": "none",
                                                "fontFamily": FONT,
                                            },
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
                                            style={"width": "130px", "fontSize": "12px"},
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "alignItems": "center",
                                        "gap": "8px",
                                        "padding": "10px 14px",
                                        "borderBottom": BORDER,
                                        "flexShrink": "0",
                                        "fontFamily": FONT,
                                    },
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
                                "background": "#3c3c3c",
                                "flexShrink": "0",
                            }
                        ),

                        # ═══ RIGHT PANEL — Chat ═══════════════════
                        html.Div(
                            [
                                # Chat header with clear button
                                html.Div(
                                    [
                                        html.Span(
                                            "CHAT",
                                            style={
                                                "fontSize": "11px",
                                                "color": TEXT_DIM,
                                                "letterSpacing": "1.5px",
                                                "fontWeight": "600",
                                            },
                                        ),
                                        html.Button(
                                            "Clear",
                                            id=f"{S}_clear_btn",
                                            n_clicks=0,
                                            style={
                                                "padding": "4px 10px",
                                                "fontSize": "11px",
                                                "fontFamily": FONT,
                                                "background": "transparent",
                                                "color": TEXT_DIM,
                                                "border": BORDER,
                                                "borderRadius": RADIUS,
                                                "cursor": "pointer",
                                                "marginLeft": "auto",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "alignItems": "center",
                                        "padding": "10px 14px",
                                        "borderBottom": BORDER,
                                        "flexShrink": "0",
                                        "fontFamily": FONT,
                                    },
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
                                        "gap": "8px",
                                        "fontFamily": FONT,
                                    },
                                ),
                                # Fixed input bar at bottom
                                html.Div(
                                    [
                                        dcc.Input(
                                            id=f"{S}_chat_input",
                                            type="text",
                                            placeholder="Ask anything about your data…",
                                            debounce=False,
                                            n_submit=0,
                                            disabled=False,
                                            style={
                                                "flex": "1",
                                                "padding": "9px 12px",
                                                "fontSize": "13px",
                                                "fontFamily": FONT,
                                                "background": BG_INPUT,
                                                "color": TEXT,
                                                "border": BORDER,
                                                "borderRight": "none",
                                                "borderRadius": f"{RADIUS} 0 0 {RADIUS}",
                                                "outline": "none",
                                            },
                                        ),
                                        html.Button(
                                            "Send",
                                            id=f"{S}_send_btn",
                                            n_clicks=0,
                                            disabled=False,
                                            style={
                                                "padding": "9px 16px",
                                                "fontSize": "13px",
                                                "fontFamily": FONT,
                                                "fontWeight": "600",
                                                "background": ACCENT,
                                                "color": "#ffffff",
                                                "border": "none",
                                                "borderRadius": f"0 {RADIUS} {RADIUS} 0",
                                                "cursor": "pointer",
                                                "whiteSpace": "nowrap",
                                                "flexShrink": "0",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "padding": "10px 14px",
                                        "borderTop": BORDER,
                                        "background": BG_PANEL,
                                        "flexShrink": "0",
                                    },
                                ),
                            ],
                            className="chat-panel",
                            style={
                                "width": "38%",
                                "minWidth": "300px",
                                "maxWidth": "500px",
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
    State("agentic_bi_page_chat_input", "value"),
    State("agentic_bi_page_store", "data"),
    prevent_initial_call=True,
)
def on_send(_clicks, _submit, user_text, store):
    if not user_text or not user_text.strip():
        return no_update, no_update, no_update, no_update, no_update, no_update

    store    = store or {"charts": [], "messages": [], "all_charts": []}
    charts   = list(store.get("charts", []))
    all_charts = list(store.get("all_charts", []))
    messages = list(store.get("messages", []))
    question = user_text.strip()

    messages.append({"role": "user", "content": question, "timestamp": uuid.uuid4().hex[:8]})

    # Guardrails
    try:
        allowed, deny_msg = _guardrails()(question)
    except Exception:
        allowed, deny_msg = True, ""

    if not allowed:
        messages.append({"role": "assistant", "content": deny_msg})
        return {**store, "charts": charts, "all_charts": all_charts, "messages": messages}, "", False, False, "Send", ""

    # Build conversation history for context
    history = [m["content"] for m in messages if m.get("role") == "user"]

    # Orchestrator
    try:
        result = _orchestrator().run(question, conversation_history=history)
    except Exception as exc:
        messages.append({"role": "assistant", "content": f"Pipeline error: {exc}"})
        return {**store, "charts": charts, "all_charts": all_charts, "messages": messages}, "", False, False, "Send", ""

    if result.get("success") and result.get("figure") is not None:
        title = (result.get("chart_spec") or {}).get("title") or "Chart"
        chart_id = str(uuid.uuid4())[:8]
        chart_data = {
            "id":          chart_id,
            "query":       question,
            "title":       title,
            "figure_json": result["figure"].to_json(),
            "sql":         result.get("sql"),
            "df_json":     result["df"].to_json(orient="records") if result.get("df") is not None else None,
            "timestamp":   uuid.uuid4().hex[:8],
        }
        charts.insert(0, chart_data)
        all_charts.insert(0, chart_data)
        reply = result.get("markdown") or f"Chart ready: **{title}**"

    elif result.get("success") and result.get("viz_failed"):
        reply = (
            "Query ran but chart could not be rendered.\n"
            + (result.get("viz_error") or "")
            + ("\n\n" + result["markdown"] if result.get("markdown") else "")
        )
    else:
        reply = result.get("error") or "Could not answer that. Try rephrasing."

    messages.append({"role": "assistant", "content": reply})
    return {**store, "charts": charts, "all_charts": all_charts, "messages": messages}, "", False, False, "Send", ""


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
    Input({"type": "delete-chart", "index": Input.ALL}, "n_clicks"),
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
    Input("agentic_bi_page_store", "data"),
    Input("agentic_bi_page_filter_query", "value"),
    Input("agentic_bi_page_sort_by", "value"),
)
def render_ui(store, filter_q, sort_by):
    store    = store or {"charts": [], "messages": [], "all_charts": []}
    all_charts = list(store.get("all_charts", []))
    charts   = list(store.get("charts", []))
    messages = list(store.get("messages", []))

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

    # ── Dashboard ───────────────────────────────────────
    if not charts:
        if all_charts and fq:
            # Charts exist but filter hides them
            empty_msg = f'No charts match "{fq}". Try a different filter.'
        else:
            # Truly empty — show suggested queries
            empty_msg = "Ask a question in the chat to generate charts."
        dash_out = [
            html.Div(
                [
                    html.Div(
                        empty_msg,
                        style={
                            "color": TEXT_DIM,
                            "fontSize": "13px",
                            "padding": "32px 8px 16px",
                            "fontFamily": FONT,
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                "Try asking:",
                                style={
                                    "color": TEXT_DIM,
                                    "fontSize": "11px",
                                    "marginBottom": "8px",
                                    "fontFamily": FONT,
                                },
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        q,
                                        id={"type": "suggested-query", "index": i},
                                        n_clicks=0,
                                        style={
                                            "padding": "6px 12px",
                                            "fontSize": "12px",
                                            "fontFamily": FONT,
                                            "background": BG_INPUT,
                                            "color": TEXT,
                                            "border": BORDER,
                                            "borderRadius": RADIUS,
                                            "cursor": "pointer",
                                            "margin": "4px",
                                        },
                                    )
                                    for i, q in enumerate(SUGGESTED_QUERIES)
                                ],
                                style={"display": "flex", "flexWrap": "wrap"},
                            ),
                        ],
                        style={"padding": "0 8px"},
                    ),
                ]
                if not all_charts
                else [
                    html.Div(
                        empty_msg,
                        style={
                            "color": TEXT_DIM,
                            "fontSize": "13px",
                            "padding": "32px 8px",
                            "fontFamily": FONT,
                        },
                    )
                ]
            )
        ]
    else:
        dash_out = []
        for c in charts:
            try:
                fig = pio.from_json(c["figure_json"])
                chart_id = c.get("id", "")
                dash_out.append(
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                c.get("title") or "Chart",
                                                style={"fontWeight": "600", "fontSize": "13px", "color": TEXT},
                                            ),
                                            html.Span(
                                                c.get("query") or "",
                                                style={"fontSize": "11px", "color": TEXT_DIM, "marginLeft": "10px"},
                                            ),
                                        ],
                                        style={"display": "flex", "alignItems": "center"},
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "⛶",
                                                id={"type": "expand-chart", "index": chart_id},
                                                n_clicks=0,
                                                title="Expand fullscreen",
                                                style={
                                                    "padding": "2px 6px",
                                                    "fontSize": "14px",
                                                    "background": "transparent",
                                                    "color": TEXT_DIM,
                                                    "border": "none",
                                                    "cursor": "pointer",
                                                },
                                            ),
                                            html.Button(
                                                "⬇",
                                                id={"type": "download-chart", "index": chart_id},
                                                n_clicks=0,
                                                title="Download as PNG",
                                                style={
                                                    "padding": "2px 6px",
                                                    "fontSize": "14px",
                                                    "background": "transparent",
                                                    "color": TEXT_DIM,
                                                    "border": "none",
                                                    "cursor": "pointer",
                                                },
                                            ),
                                            html.Button(
                                                "✕",
                                                id={"type": "delete-chart", "index": chart_id},
                                                n_clicks=0,
                                                title="Remove from dashboard",
                                                style={
                                                    "padding": "2px 6px",
                                                    "fontSize": "14px",
                                                    "background": "transparent",
                                                    "color": TEXT_DIM,
                                                    "border": "none",
                                                    "cursor": "pointer",
                                                },
                                            ),
                                        ],
                                        style={"display": "flex", "gap": "4px"},
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "justifyContent": "space-between",
                                    "alignItems": "center",
                                    "marginBottom": "8px",
                                    "fontFamily": FONT,
                                },
                            ),
                            dcc.Graph(
                                id={"type": "chart-graph", "index": chart_id},
                                figure=fig,
                                config={
                                    "displayModeBar": True,
                                    "responsive": True,
                                    "toImageButtonOptions": {
                                        "format": "png",
                                        "filename": f"chart_{chart_id}",
                                    },
                                },
                                style={"height": "300px", "minHeight": "200px"},
                            ),
                        ],
                        style={
                            "marginBottom": "16px",
                            "padding": "14px",
                            "background": BG_PANEL,
                            "borderRadius": RADIUS,
                            "border": BORDER,
                        },
                    )
                )
            except Exception:
                pass

    # ── Chat messages ────────────────────────────────────
    if not messages:
        chat_out = [
            html.Div(
                [
                    html.Div(
                        "👋 Welcome to AgenticBI",
                        style={
                            "color": TEXT,
                            "fontSize": "15px",
                            "fontWeight": "600",
                            "marginBottom": "8px",
                            "fontFamily": FONT,
                        },
                    ),
                    html.Div(
                        "Ask anything about your data and I'll generate charts and insights.",
                        style={"color": TEXT_DIM, "fontSize": "13px", "fontFamily": FONT, "marginBottom": "12px"},
                    ),
                    html.Div(
                        "Try asking:",
                        style={"color": TEXT_DIM, "fontSize": "11px", "marginBottom": "8px", "fontFamily": FONT},
                    ),
                    html.Div(
                        [
                            html.Button(
                                q,
                                id={"type": "suggested-query-chat", "index": i},
                                n_clicks=0,
                                style={
                                    "padding": "6px 12px",
                                    "fontSize": "12px",
                                    "fontFamily": FONT,
                                    "background": BG_INPUT,
                                    "color": TEXT,
                                    "border": BORDER,
                                    "borderRadius": RADIUS,
                                    "cursor": "pointer",
                                    "margin": "4px",
                                },
                            )
                            for i, q in enumerate(SUGGESTED_QUERIES)
                        ],
                        style={"display": "flex", "flexWrap": "wrap"},
                    ),
                ],
                style={"padding": "8px 0"},
            )
        ]
    else:
        chat_out = []
        for m in messages:
            is_user = m.get("role") == "user"
            content = m.get("content", "")

            # User messages: plain div; Assistant messages: dcc.Markdown for rich formatting
            if is_user:
                bubble = html.Div(
                    content,
                    style={
                        "padding": "8px 12px",
                        "borderRadius": "10px",
                        "maxWidth": "88%",
                        "whiteSpace": "pre-wrap",
                        "fontSize": "13px",
                        "lineHeight": "1.55",
                        "fontFamily": FONT,
                        "backgroundColor": BG_MSG_US,
                        "color": TEXT,
                    },
                )
            else:
                bubble = html.Div(
                    dcc.Markdown(
                        content,
                        style={
                            "fontSize": "13px",
                            "lineHeight": "1.55",
                            "fontFamily": FONT,
                            "color": TEXT,
                        },
                    ),
                    style={
                        "padding": "8px 12px",
                        "borderRadius": "10px",
                        "maxWidth": "88%",
                        "backgroundColor": BG_MSG_AI,
                    },
                )

            chat_out.append(
                html.Div(
                    bubble,
                    style={
                        "display": "flex",
                        "justifyContent": "flex-end" if is_user else "flex-start",
                    },
                )
            )

    return dash_out, chat_out


# ---------------------------------------------------------
# Callback 3 — Suggested query chips → fill input
# ---------------------------------------------------------
@callback(
    Output("agentic_bi_page_chat_input", "value", allow_duplicate=True),
    Input({"type": "suggested-query", "index": Input.ALL}, "n_clicks"),
    Input({"type": "suggested-query-chat", "index": Input.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def on_suggested_query(n_clicks_dashboard, n_clicks_chat):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    idx = json.loads(triggered)["index"]
    return SUGGESTED_QUERIES[idx]


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
