# Project Structure

## Root Directory
```
mcp-crawl4ai-rag/
├── src/                        # Main source code
├── tests/                      # Hierarchical test suite
├── scripts/                    # Utility scripts
├── docker-compose.yaml         # Docker services
├── pyproject.toml             # Project dependencies
├── CLAUDE.md                  # Development guidelines
└── README.md                  # Project documentation
```

## Source Code Organization (src/)
```
src/
├── core/                       # Core application components
│   ├── app.py                 # FastMCP server setup and tool registration
│   └── context.py             # Singleton context management
├── tools/                      # MCP tools by functionality
│   ├── web_tools.py           # Web crawling tools
│   ├── github_tools.py        # GitHub repository tools
│   ├── rag_tools.py           # Vector search tools
│   └── kg_tools.py            # Knowledge graph tools
├── services/                   # Business logic services
│   ├── embedding_service.py   # Embedding operations
│   ├── rag_service.py         # RAG pipeline
│   └── unified_indexing_service.py  # Repository processing
├── clients/                    # External service clients
│   ├── qdrant_client.py       # Vector database client
│   └── llm_api_client.py      # OpenAI/LLM API client
├── k_graph/                    # Knowledge graph system
│   ├── core/                  # Interfaces and models
│   ├── parsing/               # Multi-language parsers
│   ├── analysis/              # Code analysis and validation
│   └── services/              # High-level KG services
├── features/github/            # Modular GitHub processing
│   ├── config/                # Configuration management
│   ├── core/                  # Core models and interfaces
│   ├── discovery/             # File discovery strategies
│   ├── processors/            # Language-specific processors
│   ├── repository/            # Git operations
│   └── services/              # GitHub service layer
└── utils/                      # Utility functions
    ├── grammar_initialization.py  # Tree-sitter setup
    └── validation.py          # Data validation
```

## Test Organization (tests/)
```
tests/
├── unit/                       # Unit tests by module
│   ├── tools/                 # MCP tools tests
│   ├── services/              # Service layer tests
│   └── clients/               # Client integration tests
├── specialized/                # Domain-specific functionality
│   ├── embedding/             # Embedding system tests
│   └── knowledge_graphs/      # Knowledge graph tests
├── infrastructure/             # Infrastructure components
│   └── storage/               # Database tests
├── integration/                # End-to-end tests
└── fixtures/                   # Test data samples
```