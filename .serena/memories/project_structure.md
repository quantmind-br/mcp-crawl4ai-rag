# Project Structure

## Directory Layout

```
E:\mcp-crawl4ai-rag\
├── src/                          # Main source code
│   ├── crawl4ai_mcp.py          # Main MCP server implementation
│   └── utils.py                 # Utility functions for Supabase and embeddings
├── knowledge_graphs/            # Neo4j knowledge graph functionality
│   ├── ai_hallucination_detector.py    # Main hallucination detection script
│   ├── ai_script_analyzer.py            # Python AST analysis
│   ├── hallucination_reporter.py       # Report generation
│   ├── knowledge_graph_validator.py    # Validation against Neo4j
│   ├── parse_repo_into_neo4j.py        # GitHub repo parsing
│   ├── query_knowledge_graph.py        # Interactive graph queries
│   └── test_script.py                  # Test scripts for validation
├── PRPs/                        # Project-specific files (gitignored)
├── .serena/                     # Serena MCP configuration
├── crawled_pages.sql           # Database schema for Supabase setup
├── Dockerfile                  # Container configuration
├── pyproject.toml              # Python project configuration
├── uv.lock                     # Dependency lock file
├── .env.example                # Environment variable template
└── README.md                   # Comprehensive documentation
```

## Key Files

### Core Implementation
- **`src/crawl4ai_mcp.py`**: Main MCP server with all tools and lifespan management
- **`src/utils.py`**: Supabase integration, embedding generation, and search utilities

### Knowledge Graph System
- **`knowledge_graphs/`**: Complete Neo4j-based hallucination detection system
- **`crawled_pages.sql`**: PostgreSQL schema with pgvector extensions

### Configuration
- **`pyproject.toml`**: Dependencies and project metadata
- **`.env.example`**: Complete environment variable documentation
- **`Dockerfile`**: Production container setup with uv integration

### Documentation
- **`README.md`**: Comprehensive setup, configuration, and usage guide