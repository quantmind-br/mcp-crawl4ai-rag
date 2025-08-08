# Essential Commands

## Development Setup
```bash
# Initial setup
git clone <repository>
cd mcp-crawl4ai-rag
uv sync                    # Install dependencies

# Windows setup
setup.bat                  # Start Docker services

# Linux/Mac setup  
docker-compose up -d       # Start services

# Configuration
cp .env.example .env       # Copy environment template
# Edit .env with your API keys
```

## Running the Server
```bash
# Windows
start.bat                  # Convenience script

# Linux/Mac/Windows
uv run -m src             # Primary method (module execution)
python run_server.py     # Alternative method
```

## Testing
```bash
uv run pytest                              # Run all tests
uv run pytest tests/test_mcp_basic.py      # MCP functionality
uv run pytest tests/test_qdrant_wrapper.py # Vector database  
uv run pytest tests/integration_test.py    # Full integration
```

## Code Quality
```bash
uv run ruff check         # Linting
uv run ruff format        # Formatting
uv run ruff check --fix   # Auto-fix linting issues
```

## Docker Services
```bash
docker-compose logs qdrant    # View Qdrant logs
docker-compose logs neo4j     # View Neo4j logs
docker-compose restart        # Restart all services
docker-compose down --volumes # Stop and clean up
```

## Utilities
```bash
# Clean vector database
uv run python scripts/clean_qdrant.py

# Fix dimension mismatches  
uv run python scripts/define_qdrant_dimensions.py

# Analyze code for hallucinations
uv run python knowledge_graphs/ai_hallucination_detector.py script.py
```

## Windows System Commands
- `dir` (list files)
- `type` (view file contents)
- `mkdir` (create directory)
- `rmdir /s` (remove directory)
- `findstr` (search in files)