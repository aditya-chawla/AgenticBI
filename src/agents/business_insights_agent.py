"""
Business Insights Agent — generates human-readable Markdown narratives
from query results by combining MCP-powered statistical profiling with
LLM-based data storytelling.

Architecture:
  1. Launches the MCP Stats Server as a subprocess (stdio transport).
  2. Calls the `profile_data_statistics` tool via MCP protocol.
  3. Passes the stats + data sample + user question to the LLM.
  4. Returns a polished Markdown narrative string.
"""

import json
import os
import sys
import asyncio
import pandas as pd
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL,
    get_logger,
)

logger = get_logger("agent.business_insights")

# Path to MCP Stats Server script (same directory)
_AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_MCP_SERVER_SCRIPT = os.path.join(_AGENTS_DIR, "mcp_stats_server.py")

# ── Prompt template ──────────────────────────────────────────
INSIGHTS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a senior Business Intelligence analyst. "
        "You will be given:\n"
        "1. A user's original question about their data.\n"
        "2. A statistical profile of the query results (JSON).\n"
        "3. A sample of the actual data (first rows as markdown table).\n\n"
        "Your job is to write a clear, concise, and insightful Markdown summary "
        "that answers the user's question with supporting evidence from the data. "
        "Follow these rules:\n"
        "- Lead with the key finding or answer to the question.\n"
        "- Mention specific numbers, names, and percentages.\n"
        "- If there are notable outliers, trends, or patterns, call them out.\n"
        "- Keep the response under 150 words.\n"
        "- Do NOT include raw tables — narrate the data.\n"
        "- Use bold for emphasis on key figures.\n"
        "- If the data only has a few rows, just describe all of them briefly.\n"
        "- End with a one-sentence takeaway if appropriate."
    )),
    ("human", (
        "**User Question:** {user_question}\n\n"
        "**Statistical Profile:**\n```json\n{stats_json}\n```\n\n"
        "**Data Sample (first rows):**\n{data_sample}"
    )),
])


class BusinessInsightsAgent:
    """
    Generates Markdown narratives from query results.
    Uses MCP Stats Server for statistical profiling, then LLM for narration.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            temperature=0.3,
        )
        self.chain = INSIGHTS_PROMPT | self.llm | StrOutputParser()

    # ── MCP client call ──────────────────────────────────────
    async def _call_mcp_stats(self, data_json: str) -> str:
        """
        Launch the MCP Stats Server as a subprocess and call
        the profile_data_statistics tool via stdio transport.
        """
        python_exec = sys.executable
        server_params = StdioServerParameters(
            command=python_exec,
            args=[_MCP_SERVER_SCRIPT],
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                result = await session.call_tool(
                    "profile_data_statistics",
                    arguments={"data_json": data_json},
                )
                # MCP returns a list of content blocks; grab the text
                if result.content and len(result.content) > 0:
                    return result.content[0].text
                return "{}"

    def _get_stats_via_mcp(self, df: pd.DataFrame) -> str:
        """Synchronous wrapper around the async MCP call."""
        data_json = df.to_json(orient="records")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already inside an event loop (e.g., Dash), create a new one in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    stats_json = pool.submit(
                        asyncio.run, self._call_mcp_stats(data_json)
                    ).result(timeout=30)
            else:
                stats_json = loop.run_until_complete(self._call_mcp_stats(data_json))
        except RuntimeError:
            stats_json = asyncio.run(self._call_mcp_stats(data_json))

        return stats_json

    # ── Main entry point ─────────────────────────────────────
    def generate_narrative(self, df: pd.DataFrame, user_question: str) -> Optional[str]:
        """
        Generate a Markdown narrative for the given DataFrame + question.
        Returns None on failure (caller should fall back to raw table).
        """
        if df is None or df.empty:
            return None

        try:
            # Step 1: Get statistical profile via MCP
            logger.info("Calling MCP Stats Server for data profiling…")
            stats_json = self._get_stats_via_mcp(df)
            logger.info("MCP Stats Server returned profile (%d chars)", len(stats_json))

            # Step 2: Prepare data sample (first 5 rows as markdown)
            data_sample = df.head(5).to_markdown(index=False)

            # Step 3: Generate narrative via LLM
            logger.info("Generating narrative via LLM…")
            narrative = self.chain.invoke({
                "user_question": user_question,
                "stats_json": stats_json,
                "data_sample": data_sample,
            })

            logger.info("Narrative generated (%d chars)", len(narrative))
            return narrative.strip()

        except Exception as e:
            logger.error("BusinessInsightsAgent failed: %s", e, exc_info=True)
            return None


# ── Standalone test ──────────────────────────────────────────
if __name__ == "__main__":
    test_df = pd.DataFrame({
        "Product": ["Widget A", "Widget B", "Widget C", "Widget D", "Widget E"],
        "Revenue": [150000, 320000, 89000, 410000, 275000],
        "Units_Sold": [1200, 2800, 700, 3500, 2100],
    })

    agent = BusinessInsightsAgent()
    result = agent.generate_narrative(test_df, "Show me the top products by revenue")

    if result:
        print("\n📊 BUSINESS INSIGHTS:\n")
        print(result)
    else:
        print("❌ Failed to generate narrative.")
