# Suggested Commands

## Development Environment Setup

### Using uv (Recommended for development)
```bash
# Install uv
pip install uv

# Create and activate virtual environment
uv venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# Install dependencies
uv pip install -e .
crawl4ai-setup
```

### Using Docker (Recommended for production)
```bash
# Build Docker image
docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .

# Run with environment file
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
```

## Running the Application

### Direct Python Execution
```bash
# Run the MCP server
uv run src/crawl4ai_mcp.py
# or if in venv: python src/crawl4ai_mcp.py
```

### Environment Configuration
```bash
# Copy and configure environment
copy .env.example .env
# Edit .env with your API keys and configuration
```

## Database Setup Commands

### Supabase Setup
```bash
# In Supabase SQL Editor, run:
# Contents of crawled_pages.sql
```

### Neo4j Setup (Optional)
```bash
# For hallucination detection - clone Local AI Package
git clone https://github.com/coleam00/local-ai-packaged.git
cd local-ai-packaged
# Follow Neo4j setup instructions
```

## Knowledge Graph Commands (if enabled)

### Manual Hallucination Detection
```bash
python knowledge_graphs/ai_hallucination_detector.py [path_to_script]
```

### Interactive Graph Queries
```bash
python knowledge_graphs/query_knowledge_graph.py
```

## System Commands (Windows)

### File Operations
- `dir` - List directory contents
- `cd` - Change directory
- `type [file]` - Display file contents
- `findstr [pattern] [files]` - Search in files

### Git Operations
```bash
git status
git add .
git commit -m "message"
git push
```

## Testing and Validation

### Test MCP Server Connection
```bash
# Test if server is running
curl http://localhost:8051/health  # if SSE transport
```

### Environment Validation
```bash
# Check Python version (must be 3.12+)
python --version

# Verify uv installation
uv --version
```

## Development Workflow

1. **Setup**: Create `.env` from `.env.example`
2. **Install**: `uv pip install -e .` and `crawl4ai-setup`  
3. **Run**: `uv run src/crawl4ai_mcp.py`
4. **Test**: Connect via MCP client
5. **Develop**: Edit code and restart server