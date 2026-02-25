"""
AgenticBI Vizro frontend.
Layout: chart dashboard (left ~62%) | chat (right ~38%), Cursor-style dark UI.
"""
from __future__ import annotations

import os
import sys
import uuid
from typing import Literal

import plotly.io as pio
from dash import callback, dcc, html, Input, Output, State, no_update
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
# Custom Vizro component
# ---------------------------------------------------------
class AgenticBIPage(vm.VizroBaseModel):
    type: Literal["agentic_bi_page"] = "agentic_bi_page"

    def build(self):
        S = self.id  # id prefix

        return html.Div(
            [
                # ── Hidden state store ─────────────────────────
                dcc.Store(id=f"{S}_store", data={"charts": [], "messages": []}),

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
                                # Chat header
                                html.Div(
                                    html.Span(
                                        "CHAT",
                                        style={
                                            "fontSize": "11px",
                                            "color": TEXT_DIM,
                                            "letterSpacing": "1.5px",
                                            "fontWeight": "600",
                                        },
                                    ),
                                    style={
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
    Input("agentic_bi_page_send_btn", "n_clicks"),
    Input("agentic_bi_page_chat_input", "n_submit"),
    State("agentic_bi_page_chat_input", "value"),
    State("agentic_bi_page_store", "data"),
    prevent_initial_call=True,
)
def on_send(_clicks, _submit, user_text, store):
    if not user_text or not user_text.strip():
        return no_update, no_update

    store    = store or {"charts": [], "messages": []}
    charts   = list(store.get("charts", []))
    messages = list(store.get("messages", []))
    question = user_text.strip()

    messages.append({"role": "user", "content": question})

    # Guardrails
    try:
        allowed, deny_msg = _guardrails()(question)
    except Exception:
        allowed, deny_msg = True, ""

    if not allowed:
        messages.append({"role": "assistant", "content": deny_msg})
        return {**store, "charts": charts, "messages": messages}, ""

    # Orchestrator
    try:
        result = _orchestrator().run(question)
    except Exception as exc:
        messages.append({"role": "assistant", "content": f"Pipeline error: {exc}"})
        return {**store, "charts": charts, "messages": messages}, ""

    if result.get("success") and result.get("figure") is not None:
        title = (result.get("chart_spec") or {}).get("title") or "Chart"
        charts.insert(0, {
            "id":          str(uuid.uuid4())[:8],
            "query":       question,
            "title":       title,
            "figure_json": result["figure"].to_json(),
        })
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
    return {**store, "charts": charts, "messages": messages}, ""


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
    store    = store or {"charts": [], "messages": []}
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
        dash_out = [
            html.Div(
                "Ask a question in the chat to generate charts.",
                style={
                    "color": TEXT_DIM,
                    "fontSize": "13px",
                    "padding": "32px 8px",
                    "fontFamily": FONT,
                },
            )
        ]
    else:
        dash_out = []
        for c in charts:
            try:
                fig = pio.from_json(c["figure_json"])
                dash_out.append(
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
                                style={"marginBottom": "8px", "fontFamily": FONT},
                            ),
                            dcc.Graph(
                                figure=fig,
                                config={"displayModeBar": True, "responsive": True},
                                style={"height": "300px"},
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
                "Ask anything about your data…",
                style={"color": TEXT_DIM, "fontSize": "13px", "fontFamily": FONT},
            )
        ]
    else:
        chat_out = []
        for m in messages:
            is_user = m.get("role") == "user"
            chat_out.append(
                html.Div(
                    html.Div(
                        m.get("content", ""),
                        style={
                            "padding": "8px 12px",
                            "borderRadius": "10px",
                            "maxWidth": "88%",
                            "whiteSpace": "pre-wrap",
                            "fontSize": "13px",
                            "lineHeight": "1.55",
                            "fontFamily": FONT,
                            "backgroundColor": BG_MSG_US if is_user else BG_MSG_AI,
                            "color": TEXT,
                        },
                    ),
                    style={
                        "display": "flex",
                        "justifyContent": "flex-end" if is_user else "flex-start",
                    },
                )
            )

    return dash_out, chat_out


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
