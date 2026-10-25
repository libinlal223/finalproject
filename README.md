# VADER — Hybrid Multi-Agent AI System with Adversarial Collaborative Reasoning

> Final-Year CSE Project

## Overview

VADER is a web application where multiple LLM-powered agents independently generate answers, challenge incorrect or unsupported claims, revise their answers when necessary, and produce a final response through a Judge agent.

## Agent Roles

| Agent | Role |
|-------|------|
| **Planner** | Decomposes user queries into structured subtasks |
| **Executor A** | Independently generates a candidate answer |
| **Executor B** | Independently generates an alternative candidate answer |
| **Critic** | Identifies errors, contradictions, omissions; retrieves evidence |
| **Judge** | Synthesises the final answer from all inputs |

## Technology Stack

- **Frontend**: React + Vite (TypeScript)
- **Backend**: Python 3.13 + FastAPI
- **LLM Providers**: Groq API, Google Gemini API (mock adapter for local dev)
- **Validation**: Pydantic v2
- **Testing**: pytest + pytest-asyncio
- **Evaluation**: NumPy, Pandas

## Quick Start

### Backend

```bash
cd backend

# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate    # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# 3. Configure environment
copy .env.example .env        # Windows
# cp .env.example .env        # Linux/Mac
# Edit .env and set your API keys

# 4. Run the development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- Swagger UI: http://localhost:8000/api/docs
- Health check: http://localhost:8000/api/health

### Running Tests (no API keys required)

```bash
cd backend
.\venv\Scripts\activate
pytest
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Environment Variables

See [`backend/.env.example`](backend/.env.example) for all configurable options.

**Never commit `.env` to version control.** Use `LLM_PROVIDER=mock` for local development without API keys.

## Project Structure

```
vader/
├── backend/
│   ├── app/
│   │   ├── config.py          # Pydantic-settings configuration
│   │   ├── main.py            # FastAPI application factory
│   │   ├── models/schemas.py  # Request/response schemas
│   │   ├── llm/               # LLM adapter interface + providers
│   │   └── routers/           # API route handlers
│   └── tests/                 # pytest test suite
└── frontend/                  # React + Vite frontend (Phase 6)
```

## Implementation Phases

- [x] **Phase 1** — Backend foundation (FastAPI, config, schemas, LLM interface, health endpoint, tests)
- [ ] **Phase 2** — LLM adapters (Groq, Gemini) + full schema suite
- [ ] **Phase 3** — Agent implementations (Planner, Executor A/B, Critic, Judge)
- [ ] **Phase 4** — Debate orchestrator + evidence retrieval
- [ ] **Phase 5** — API routers + SSE streaming
- [ ] **Phase 6** — React frontend
- [ ] **Phase 7** — Evaluation module + baseline
- [ ] **Phase 8** — Integration, hardening, polish

## Disclaimer

Multiple agents do not guarantee correctness. VADER is a research demonstration of adversarial collaborative reasoning, not a production fact-checking system.
