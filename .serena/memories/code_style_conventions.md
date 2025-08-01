# Code Style and Conventions

## Python Style Guidelines

### General Conventions
- **Python Version**: 3.12+ with modern async/await patterns
- **Import Style**: Standard library first, then third-party, then local imports
- **Async/Await**: Consistent use of async functions with proper await syntax
- **Type Hints**: Optional typing used (e.g., `Optional[Any]`)

### Code Organization
- **Dataclasses**: Used for structured data (`@dataclass` decorator)
- **Context Management**: Lifespan context pattern for dependency injection
- **Tool Decorators**: `@mcp.tool()` decorator for MCP tool functions
- **Error Handling**: Try-catch with JSON response formatting

### Function Structure
- **Docstrings**: Google-style docstrings with Args/Returns sections
- **Parameter Types**: Context object first, then typed parameters
- **Return Format**: JSON strings for tool responses
- **Async Functions**: Consistent async/await usage throughout

### Naming Conventions
- **Functions**: snake_case (e.g., `crawl_single_page`, `smart_chunk_markdown`)
- **Classes**: PascalCase (e.g., `Crawl4AIContext`)
- **Variables**: snake_case with descriptive names
- **Constants**: Environment variables in UPPER_CASE
- **Private Functions**: Leading underscore (e.g., `_handle_repos_command`)

### Data Handling
- **JSON Responses**: Pretty-printed with `indent=2`
- **Metadata**: Dictionary-based with consistent key naming
- **Error Format**: Structured JSON with success/error fields
- **URL Parsing**: `urlparse` for consistent URL handling

### Comments and Documentation
- **Tool Descriptions**: Comprehensive docstrings explaining purpose and usage
- **Inline Comments**: Minimal, focus on complex logic explanation
- **Code Organization**: Logical grouping with clear separation