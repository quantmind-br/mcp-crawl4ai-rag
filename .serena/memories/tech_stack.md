# Tech Stack

## Core Technologies

### Python Environment
- **Python**: 3.12+ (required)
- **Package Manager**: `uv` (preferred) or `pip`
- **Virtual Environment**: `.venv` (when using uv directly)

### Main Dependencies
- **crawl4ai==0.6.2**: Web crawling engine
- **mcp==1.7.1**: Model Context Protocol implementation
- **supabase==2.15.1**: Vector database client
- **openai==1.71.0**: Embeddings and LLM API
- **sentence-transformers>=4.1.0**: Local reranking models
- **neo4j>=5.28.1**: Knowledge graph database
- **dotenv==0.9.9**: Environment configuration

### Infrastructure
- **Vector Database**: Supabase (PostgreSQL with pgvector)
- **Knowledge Graph**: Neo4j (optional, for hallucination detection)
- **Container**: Docker (recommended deployment)
- **Transport**: SSE (Server-Sent Events) or stdio

### External Services
- **OpenAI API**: For embeddings (text-embedding-3-small) and LLM calls
- **Supabase**: Hosted PostgreSQL with vector capabilities
- **Neo4j**: Local or cloud (AuraDB) graph database

## Architecture Pattern
- **MCP Server**: Async Python server following MCP protocol
- **Context Management**: Dependency injection via lifespan context
- **Tool-based**: Decorated functions as MCP tools
- **Modular RAG**: Configurable strategies via environment variables