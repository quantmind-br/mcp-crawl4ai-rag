# Project Structure

## Root Directory
- `src/`: Main application code
- `tests/`: Test suites and fixtures
- `knowledge_graphs/`: Neo4j integration and analysis tools
- `scripts/`: Utility scripts for maintenance
- `PRPs/`: Project Requirement Packages documentation
- `pyproject.toml`: Project configuration and dependencies
- `docker-compose.yaml`: Service orchestration
- `run_server.py`: Alternative entry point

## Source Code Structure (`src/`)
```
src/
├── __main__.py           # Module entry point
├── core/                 # Core application logic
│   ├── app.py           # Main server application
│   └── context.py       # Application context
├── clients/             # External service clients
│   ├── llm_api_client.py
│   └── qdrant_client.py
├── services/            # Business logic services
│   ├── embedding_service.py
│   └── rag_service.py
├── tools/               # MCP tool implementations
│   ├── web_tools.py     # Web crawling tools
│   ├── rag_tools.py     # RAG search tools
│   ├── github_tools.py  # GitHub integration
│   └── kg_tools.py      # Knowledge graph tools
├── features/            # Feature implementations
│   └── github_processor.py
├── utils/               # Utility modules
│   ├── validation.py
│   └── grammar_initialization.py
└── [various config files]
```

## Entry Points
- **Primary**: `uv run -m src` (module execution)
- **Alternative**: `python run_server.py` (direct script)
- **Windows**: `start.bat` (convenience script)

## Configuration Files
- `.env`: Environment variables (copied from `.env.example`)
- `pyproject.toml`: Dependencies and project metadata
- `docker-compose.yaml`: Docker services configuration