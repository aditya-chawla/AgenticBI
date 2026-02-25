# AgenticBI – Local Setup Guide

This guide explains what you need and how to get the project running on your machine.

---

## What You Need (Requirements Overview)

### 1. **Software & Tools**

| Requirement | Purpose |
|-------------|--------|
| **Python 3.10+** | Run the agents (LangChain, LangGraph, Vizro, etc.) |
| **Docker Desktop** | Run PostgreSQL (AdventureWorks) and optionally pgAdmin |
| **Ollama** | Local LLM for NL2SQL, SQL correction, and chart suggestions (uses `llama3`) |
| **Git** | Clone/update the repo (you already have the project) |

### 2. **Files in the Repo**

- **`requirements.txt`** – Python dependencies (LangChain, ChromaDB, Vizro, psycopg2, etc.)
- **`docker-compose.yml`** – Defines Postgres (AdventureWorks) + pgAdmin
- **`src/agents/config.py`** – DB credentials, LLM model name, vector DB path (no `.env` required by default)
- **`src/agents/chroma_db_data/`** – Created when you run schema ingestion; stores the vector index (gitignored)

### 3. **VM / Machine**

- **OS:** Windows, macOS, or Linux.
- **RAM:** 8 GB minimum; 16 GB+ recommended (Ollama + ChromaDB + Postgres).
- **Docker:** Must be running for Postgres. No VM required if using Docker Desktop.
- **Disk:** ~2–3 GB for Docker images, Python venv, and Ollama model.

### 4. **No Mandatory `.env`**

The app uses defaults in `config.py` (e.g. `localhost:5432`, password `password`). You can override with environment variables later if needed.

---

## Step-by-Step Setup

### Step 1: Install Python 3.10+

- **Windows:** [python.org/downloads](https://www.python.org/downloads/) or `winget install Python.Python.3.12`
- **macOS:** `brew install python@3.12`
- **Linux:** `sudo apt install python3.10 python3.10-venv` (or equivalent)

Verify:

```bash
python --version
# or
python3 --version
```

### Step 2: Install Docker Desktop

- **Windows/macOS:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Linux:** Docker Engine + Docker Compose

Ensure Docker is **running** (you’ll start containers in the next step).

### Step 3: Start the Database

From the **project root** (`AgenticBI`):

```bash
docker-compose up -d
```

This starts:

- **Postgres** with AdventureWorks on `localhost:5432` (user: `postgres`, password: `password`)
- **pgAdmin** (optional) on `http://localhost:8080` (admin@admin.com / admin)

Check that Postgres is up:

```bash
docker ps
# You should see local_adventureworks (and optionally pgadmin_gui)
```

### Step 4: Install Ollama and Pull `llama3`

1. Install: [ollama.com](https://ollama.com) (Windows/macOS/Linux).
2. Pull the model used in the project:

```bash
ollama pull llama3
```

3. Keep Ollama running (it’s the local LLM server). Verify:

```bash
ollama list
# Should list llama3
```

### Step 5: Create a Virtual Environment and Install Dependencies

In the project root:

**Windows (PowerShell):**

```powershell
cd "c:\Users\adity\Documents\projects\agentic bi\AgenticBI"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS/Linux:**

```bash
cd /path/to/AgenticBI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First install can take a few minutes (sentence-transformers, ChromaDB, etc.).

### Step 6: Build the Schema Index (ChromaDB)

The NL2SQL agent needs a vector index of the database schema. Run the schema ingestion agent **once** (with Postgres and venv active):

**Windows (PowerShell):**

```powershell
cd src\agents
python -m schema_ingestion_agent
```

**macOS/Linux:**

```bash
cd src/agents
python -m schema_ingestion_agent
```

Or from project root:

```bash
python -m src.agents.schema_ingestion_agent
```

You should see logs and a new `src/agents/chroma_db_data/` folder. This step only needs to be repeated if the DB schema changes.

### Step 7: Run the Vizro Frontend (Dashboard + Chat)

From **project root**, with venv activated:

**Windows:**

```powershell
cd "c:\Users\adity\Documents\projects\agentic bi\AgenticBI"
.\.venv\Scripts\Activate.ps1
python -m app.main
```

**macOS/Linux:** from project root with venv active, run `python -m app.main`

Then open **http://localhost:8050** in your browser. Use the chat on the right; charts appear on the left dashboard. Top menu: filter/sort charts by query. “Show me the top 5 employees by vacation hours.”

### Step 8: Run the Orchestrator from CLI (Optional)

From project root: `cd src/agents` then `python -m orchestrator_agent` for a CLI pipeline test.

### Step 9 (Optional): Run Tests

From project root:

```bash
# Activate venv first, then:
cd tests
python -m pytest test_orchestrator_agent.py -v
# Or run a single test:
python test_orchestrator_agent.py
```

Tests expect Postgres, Ollama (llama3), and the ChromaDB schema index to be ready.

---

## Run with Docker (App Included)

You can run the **database + Vizro app** in Docker. Ollama must still run on the **host** (not in Docker).

1. **Start DB + app:** `docker-compose up -d` (starts db, pgadmin, and app on http://localhost:8050).
2. **Build the schema index once** inside the app container (must run from agents dir so `config` is found):  
   `docker-compose exec app sh -c "cd /app/src/agents && python -m schema_ingestion_agent"`
3. **Ensure Ollama is running on the host** and has `ollama pull llama3`.
4. Open **http://localhost:8050** and use the chat. The app uses `OLLAMA_HOST=http://host.docker.internal:11434` to reach your local Ollama.

---

## Quick Checklist

- [ ] Python 3.10+ installed
- [ ] Docker Desktop installed and running
- [ ] `docker-compose up -d` (Postgres + pgAdmin + app)
- [ ] Ollama installed and `ollama pull llama3` (Ollama running)
- [ ] Venv created and `pip install -r requirements.txt` (for local run)
- [ ] Schema index built: `python -m schema_ingestion_agent` from `src/agents` (or via `docker-compose exec app ...` when using Docker)
- [ ] Vizro app: `python -m app.main` then http://localhost:8050

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| **Connection refused to localhost:5432** | Docker running? `docker-compose up -d` and `docker ps` |
| **Ollama / LLM errors** | Is Ollama running? `ollama list` and `ollama run llama3` once |
| **ChromaDB / “no schema”** | Run schema ingestion from `src/agents`: `python -m schema_ingestion_agent` |
| **Import errors** | Run from `src/agents` or set `PYTHONPATH` to project root; venv activated |
| **Apple Silicon (M1/M2/M3)** | `docker-compose.yml` uses `platform: linux/amd64` for Postgres; should work |

---

## Summary

You need **Python**, **Docker** (for Postgres and optionally the app), and **Ollama** (with `llama3`) on the host. No `.env` is required for the default setup. After installing deps and building the schema index once, run **`python -m app.main`** and open http://localhost:8050 for the Vizro dashboard and chat. Or use **Docker** to run the app with `docker-compose up -d` and build the schema inside the container once.
