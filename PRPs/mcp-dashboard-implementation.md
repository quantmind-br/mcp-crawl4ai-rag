# MCP Server Management Dashboard - BASE PRP

## Goal

**Feature Goal**: Create a comprehensive web dashboard that eliminates the complexity of .env file management and provides standalone data ingestion capabilities for the MCP (Model Context Protocol) server, enabling developers to configure and operate their AI-powered crawling and RAG systems through a modern web interface.

**Deliverable**: Production-ready React Admin dashboard with FastAPI backend providing:
1. Visual configuration management for all 156+ environment variables
2. Standalone execution of all 12 MCP tools without LLM agent dependency
3. Real-time job monitoring with WebSocket progress tracking
4. System health monitoring with live metrics

**Success Definition**: 
- Dashboard successfully manages all MCP server configurations with validation
- All MCP tools execute through web interface with real-time progress tracking
- System metrics display live health status of all services (Qdrant, Neo4j, Redis)
- 80% reduction in setup time compared to manual .env editing
- All validation tests pass: linting, type checking, unit tests, integration tests

## User Persona

**Target User**: Software developers implementing and maintaining MCP servers in their projects

**Use Case**: Primary scenarios include:
1. Initial MCP server setup and configuration
2. Daily configuration management and feature flag adjustments
3. Data ingestion job execution and monitoring
4. System health monitoring and troubleshooting

**User Journey**: 
1. Access dashboard → 2. Configure settings via visual forms → 3. Execute data ingestion jobs → 4. Monitor progress and system health → 5. Review results and logs

**Pain Points Addressed**:
- Manual .env file editing (156+ variables across 10 categories)
- No visibility into data ingestion job progress
- Difficult troubleshooting without centralized monitoring
- Complex multi-service dependency management

## Why

- **Developer Experience**: Eliminates error-prone manual configuration editing and provides intuitive visual interface
- **Operational Efficiency**: Enables standalone data ingestion without requiring LLM agent connections
- **System Observability**: Provides real-time monitoring and health checking for all service dependencies
- **Reduced Time to Value**: 80% faster initial setup and configuration management
- **Error Prevention**: Built-in validation prevents configuration errors that cause service failures

## What

A modern web application consisting of:

### Frontend (React Admin Dashboard)
- Configuration management interface with categorized forms and validation
- Real-time job queue with progress tracking and WebSocket updates
- System metrics dashboard with live charts and service health indicators
- Job execution interface for all 12 MCP tools with parameter forms

### Backend (FastAPI Web Service)
- REST API for configuration management with .env file synchronization
- WebSocket endpoints for real-time job progress updates
- Job queue management with background task processing
- System health monitoring with database connection checking
- MCP tool proxy for executing existing tools through HTTP interface

### Integration Layer
- Shared database access (Qdrant, Neo4j, Redis) with existing MCP server
- Configuration synchronization between dashboard and MCP server
- Real-time communication via WebSocket for live updates

### Success Criteria

- [ ] All 156+ environment variables manageable through visual forms with validation
- [ ] All 12 MCP tools executable through web interface with real-time progress
- [ ] WebSocket real-time updates for job progress and system metrics
- [ ] System health dashboard showing Qdrant, Neo4j, Redis service status
- [ ] Configuration export/import functionality with .env file sync
- [ ] Job history and results tracking with detailed logging
- [ ] Responsive design supporting desktop and mobile access
- [ ] Complete test coverage: unit, integration, and end-to-end tests

## All Needed Context

### Context Completeness Check

_"If someone knew nothing about this codebase, would they have everything needed to implement this successfully?"_

**YES** - This PRP provides:
- Exact file patterns and implementation guidance from existing codebase analysis
- Specific library integration patterns with code examples
- Complete data model patterns and validation approaches
- Precise database sharing strategies and connection management
- Detailed WebSocket implementation patterns for real-time updates
- Comprehensive testing strategies following existing patterns

### Documentation & References

```yaml
# CRITICAL FastAPI + MCP Integration Patterns
- docfile: PRPs/ai_docs/fastapi_mcp_integration_patterns.md
  why: Essential patterns for creating parallel FastAPI service that shares databases with existing MCP server
  section: "Database Sharing Pattern, MCP Tool Proxy Pattern, WebSocket Real-Time Updates"
  critical: DO NOT modify existing MCP server - create parallel service with shared database access

# ESSENTIAL React Admin Dashboard Implementation
- docfile: PRPs/ai_docs/react_admin_dashboard_patterns.md  
  why: Complete React Admin framework integration with custom data providers and real-time components
  section: "Configuration Management Interface, Job Management Interface, Real-time Monitoring Dashboard"
  critical: Use React Admin framework for rapid development with Material-UI components

# MANDATORY WebSocket Job Management Patterns
- docfile: PRPs/ai_docs/websocket_job_management_patterns.md
  why: Production-ready WebSocket patterns for job progress tracking and real-time updates
  section: "Robust Connection Manager, Job Progress Tracking System, Frontend WebSocket Integration"
  critical: Implement proper connection management with ping/pong, error recovery, and message batching

# Existing MCP Server Architecture (DO NOT MODIFY)
- file: src/core/app.py
  why: FastMCP server structure and tool registration patterns to understand for integration
  pattern: FastMCP application factory, tool registration, context management
  gotcha: Uses SSE transport on port 8051, not HTTP REST - dashboard must be separate service

# Database Client Patterns (REUSE THESE)
- file: src/clients/qdrant_client.py
  why: QdrantClientWrapper pattern for database connections
  pattern: Connection singleton, environment-based configuration
  gotcha: Must reuse existing client patterns for database access consistency

# Service Layer Architecture (FOLLOW THIS)
- file: src/services/rag_service.py
  why: Service class structure and dependency injection patterns
  pattern: Service class with async methods, error handling, logging
  gotcha: All services use async patterns - maintain consistency

# Configuration Management (MIRROR THIS)
- file: .env.example
  why: Complete list of 156+ environment variables organized by category
  pattern: Category organization, feature flags, performance settings
  gotcha: Must handle sensitive variables (API keys) with proper masking

# Testing Infrastructure (FOLLOW EXACTLY)
- file: tests/conftest.py
  why: Pytest fixture patterns and test organization structure
  pattern: Mock fixtures, test environment setup, test data management
  gotcha: Tests organized hierarchically - dashboard tests go in tests/unit/dashboard/

# Job Execution Patterns (REFERENCE FOR PROXY)
- file: src/tools/web_tools.py
  why: Tool execution patterns and progress tracking
  pattern: Async tool functions, context management, error handling
  gotcha: Tools expect specific context structure - proxy must mock correctly

# Data Models (FOLLOW CONVENTIONS)
- file: src/models/unified_indexing_models.py  
  why: Dataclass patterns for data models and validation
  pattern: Dataclass with __post_init__ validation, field defaults
  gotcha: Uses dataclasses, not Pydantic - maintain consistency

# Performance Configuration (REUSE SETTINGS)
- file: src/utils/performance_config.py
  why: Performance settings and resource management patterns
  pattern: CPU/IO worker configuration, batch sizes, concurrent limits
  gotcha: Performance settings affect both MCP server and dashboard - must coordinate

# Windows Unicode Compatibility (CRITICAL)
- file: CLAUDE.md
  why: Unicode character guidelines for Windows console compatibility
  pattern: ASCII-only output, no emojis or special characters in logs
  gotcha: UnicodeEncodeError will break development workflow - strictly ASCII only
```

### Current Codebase Tree

```
mcp-crawl4ai-rag/
├── src/                          # Main application code
│   ├── core/                     # Core MCP server (DO NOT MODIFY)
│   │   ├── app.py               # FastMCP server setup
│   │   └── context.py           # Singleton context management
│   ├── tools/                    # MCP tools (12 tools to proxy)
│   │   ├── web_tools.py         # Web crawling tools
│   │   ├── github_tools.py      # GitHub repository tools
│   │   ├── rag_tools.py         # RAG query tools
│   │   └── kg_tools.py          # Knowledge graph tools
│   ├── services/                 # Business logic services (REUSE)
│   │   ├── rag_service.py       # RAG operations
│   │   └── unified_indexing_service.py
│   ├── clients/                  # Database clients (REUSE)
│   │   └── qdrant_client.py     # Qdrant connection wrapper
│   └── utils/                    # Utility functions
├── tests/                        # Test infrastructure (FOLLOW STRUCTURE)
│   ├── unit/                     # Unit tests by module
│   ├── integration/              # Integration tests
│   └── conftest.py              # Pytest fixtures
├── PRPs/ai_docs/                # Implementation guidance
└── docker-compose.yaml          # Database services
```

### Desired Codebase Tree with Dashboard Files

```
mcp-crawl4ai-rag/
├── dashboard/                    # NEW: Dashboard application
│   ├── backend/                  # FastAPI backend service
│   │   ├── api/
│   │   │   ├── v1/
│   │   │   │   ├── endpoints/
│   │   │   │   │   ├── config.py      # Configuration management
│   │   │   │   │   ├── jobs.py        # Job management
│   │   │   │   │   ├── metrics.py     # System metrics
│   │   │   │   │   └── websocket.py   # WebSocket endpoints
│   │   │   │   └── api.py             # API router
│   │   │   └── dependencies.py        # Dependency injection
│   │   ├── core/
│   │   │   ├── config.py              # Dashboard settings
│   │   │   ├── context.py             # Dashboard context
│   │   │   └── security.py            # Authentication
│   │   ├── jobs/
│   │   │   ├── manager.py             # Job queue management
│   │   │   ├── executor.py            # Job execution with progress
│   │   │   └── progress_tracker.py    # Progress tracking
│   │   ├── models/
│   │   │   ├── config_models.py       # Configuration data models
│   │   │   ├── job_models.py          # Job management models
│   │   │   └── metric_models.py       # Metrics models
│   │   ├── services/
│   │   │   ├── config_service.py      # Configuration management
│   │   │   ├── mcp_proxy.py           # MCP tool proxy service
│   │   │   └── metrics_service.py     # System monitoring
│   │   ├── websocket/
│   │   │   └── manager.py             # WebSocket connection management
│   │   └── main.py                    # FastAPI application
│   └── frontend/                      # React Admin frontend
│       ├── src/
│       │   ├── components/
│       │   │   ├── configuration/     # Config management UI
│       │   │   ├── jobs/              # Job queue UI
│       │   │   ├── metrics/           # System metrics UI
│       │   │   └── common/            # Shared components
│       │   ├── hooks/
│       │   │   ├── useWebSocket.js    # WebSocket management
│       │   │   └── useJobProgress.js  # Job progress tracking
│       │   ├── services/
│       │   │   └── api.js             # API client
│       │   ├── dataProvider.js        # React Admin data provider
│       │   └── App.js                 # Main application
│       ├── package.json
│       └── public/
└── tests/unit/dashboard/              # Dashboard tests
    ├── backend/                       # Backend unit tests
    └── frontend/                      # Frontend unit tests
```

### Known Gotchas & Library Quirks

```python
# CRITICAL: MCP Server Integration
# The existing MCP server uses FastMCP framework on port 8051 with SSE transport
# Dashboard MUST be a separate FastAPI service on different port (8080)
# DO NOT modify existing MCP server code - create parallel service

# CRITICAL: Database Connection Sharing
# Reuse existing QdrantClientWrapper and service classes
# from src.clients.qdrant_client import QdrantClientWrapper
# from src.services.rag_service import RagService
# All database connections must use same environment variables

# CRITICAL: Windows Unicode Compatibility
# NEVER use Unicode characters (emojis, special symbols) in logs or output
# Use ASCII alternatives only: "SUCCESS" not "✅", "ERROR" not "❌"
# UnicodeEncodeError will break Windows development workflow

# FastAPI Context Management
# Use dependency injection with @lru_cache for singletons
# Async context managers for database connections
# Proper lifespan events for resource management

# React Admin Data Provider
# Must implement all CRUD operations (getList, getOne, update, create, delete)
# Handle pagination, sorting, filtering in API endpoints
# Use React Admin conventions for data structure

# WebSocket Connection Management
# Implement ping/pong for connection health
# Handle reconnection with exponential backoff
# Proper cleanup on component unmount
# Rate limiting for message frequency

# Job Queue Management
# Use asyncio.create_task for background job execution
# Implement progress callbacks for real-time updates
# Store job state in memory or Redis for persistence
# Handle job cancellation and error recovery

# Material-UI Theme Integration
# Use React Admin's built-in theming system
# Consistent component styling with Material-UI v5
# Responsive design for mobile compatibility
```

## Implementation Blueprint

### Data Models and Structure

Create data models following existing dataclass patterns for type safety and consistency.

```python
# dashboard/backend/models/config_models.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum

class ConfigCategory(str, Enum):
    """Configuration categories matching .env structure"""
    SERVER = "server"
    AI_MODELS = "ai_models"
    RAG_FEATURES = "rag_features"
    DATABASES = "databases"
    PERFORMANCE = "performance"
    CACHE = "cache"

@dataclass
class ConfigSetting:
    """Individual configuration setting"""
    key: str
    value: str
    category: ConfigCategory
    description: str
    data_type: str = "string"  # string, number, boolean, select
    options: List[str] = field(default_factory=list)
    sensitive: bool = False
    validation_regex: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration setting"""
        if not self.key:
            raise ValueError("Configuration key is required")
        if self.data_type not in ["string", "number", "boolean", "select"]:
            raise ValueError(f"Invalid data type: {self.data_type}")

@dataclass
class ConfigUpdate:
    """Configuration update request"""
    changes: Dict[str, str]
    restart_required: bool = False
    
    def __post_init__(self):
        """Validate configuration changes"""
        if not self.changes:
            raise ValueError("No configuration changes provided")

# dashboard/backend/models/job_models.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
import uuid

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class JobRequest:
    """Job execution request"""
    tool_name: str
    parameters: Dict[str, Any]
    client_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate job request"""
        if not self.tool_name:
            raise ValueError("Tool name is required")
        
        # Validate tool name against known MCP tools
        valid_tools = {
            "crawl_single_page", "smart_crawl_url", "smart_crawl_github",
            "perform_rag_query", "get_available_sources", "search_code_examples",
            "parse_github_repository", "check_ai_script_hallucinations", 
            "query_knowledge_graph"
        }
        if self.tool_name not in valid_tools:
            raise ValueError(f"Unknown tool: {self.tool_name}")

@dataclass
class Job:
    """Job execution state"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    client_id: Optional[str] = None
```

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE dashboard/backend/core/config.py
  - IMPLEMENT: DashboardSettings with Pydantic BaseSettings
  - FOLLOW pattern: Environment-based configuration from existing .env structure
  - NAMING: DashboardSettings class, use same env var names as MCP server
  - PLACEMENT: Core configuration in dashboard/backend/core/

Task 2: CREATE dashboard/backend/core/context.py  
  - IMPLEMENT: DashboardContext dataclass reusing existing MCP context
  - FOLLOW pattern: src/core/context.py singleton pattern and resource management
  - NAMING: DashboardContext class, async def create() method
  - DEPENDENCIES: Import from src.core.context import Crawl4AIContext
  - PLACEMENT: Context management in dashboard/backend/core/

Task 3: CREATE dashboard/backend/models/config_models.py
  - IMPLEMENT: ConfigSetting, ConfigUpdate, ConfigCategory dataclasses
  - FOLLOW pattern: src/models/unified_indexing_models.py dataclass structure
  - NAMING: CamelCase classes, snake_case fields, __post_init__ validation
  - PLACEMENT: Data models in dashboard/backend/models/

Task 4: CREATE dashboard/backend/models/job_models.py
  - IMPLEMENT: JobRequest, Job, JobStatus dataclasses and enums
  - FOLLOW pattern: src/models/unified_indexing_models.py validation patterns
  - NAMING: Job-related models with status enum
  - DEPENDENCIES: datetime, uuid, enum imports
  - PLACEMENT: Job models in dashboard/backend/models/

Task 5: CREATE dashboard/backend/services/mcp_proxy.py
  - IMPLEMENT: MCPToolProxy class with execute_tool method
  - FOLLOW pattern: src/services/rag_service.py async service pattern
  - NAMING: MCPToolProxy class, async def execute_tool method
  - DEPENDENCIES: Import tool functions from src.tools modules
  - PLACEMENT: Service layer in dashboard/backend/services/

Task 6: CREATE dashboard/backend/websocket/manager.py
  - IMPLEMENT: DashboardWebSocketManager with connection pooling
  - FOLLOW pattern: AI docs websocket_job_management_patterns.md connection manager
  - NAMING: DashboardWebSocketManager class, async methods
  - FEATURES: Connection health monitoring, job subscriptions, broadcast
  - PLACEMENT: WebSocket management in dashboard/backend/websocket/

Task 7: CREATE dashboard/backend/jobs/progress_tracker.py
  - IMPLEMENT: JobProgressTracker with stage management
  - FOLLOW pattern: AI docs websocket_job_management_patterns.md progress tracker
  - NAMING: JobProgressTracker class, ProgressStage enum
  - FEATURES: Progress callbacks, stage transitions, error tracking
  - PLACEMENT: Job management in dashboard/backend/jobs/

Task 8: CREATE dashboard/backend/jobs/manager.py
  - IMPLEMENT: DashboardJobManager with queue management
  - FOLLOW pattern: Async job execution with progress tracking
  - NAMING: DashboardJobManager class, async def submit_job method
  - DEPENDENCIES: progress_tracker, websocket manager, mcp_proxy
  - PLACEMENT: Job management in dashboard/backend/jobs/

Task 9: CREATE dashboard/backend/services/config_service.py
  - IMPLEMENT: ConfigurationService with .env file management
  - FOLLOW pattern: src/services/rag_service.py service structure
  - NAMING: ConfigurationService class, async CRUD methods
  - FEATURES: Validate, read, write .env file, category organization
  - PLACEMENT: Service layer in dashboard/backend/services/

Task 10: CREATE dashboard/backend/api/v1/endpoints/config.py
  - IMPLEMENT: FastAPI endpoints for configuration management
  - FOLLOW pattern: FastAPI router with dependency injection
  - NAMING: Router with /config prefix, CRUD operation endpoints
  - DEPENDENCIES: ConfigurationService, Pydantic models
  - PLACEMENT: API endpoints in dashboard/backend/api/v1/endpoints/

Task 11: CREATE dashboard/backend/api/v1/endpoints/jobs.py
  - IMPLEMENT: FastAPI endpoints for job management
  - FOLLOW pattern: FastAPI async endpoints with proper error handling
  - NAMING: Router with /jobs prefix, job CRUD operations
  - DEPENDENCIES: DashboardJobManager, job models
  - PLACEMENT: API endpoints in dashboard/backend/api/v1/endpoints/

Task 12: CREATE dashboard/backend/api/v1/endpoints/websocket.py
  - IMPLEMENT: WebSocket endpoints for real-time updates
  - FOLLOW pattern: FastAPI WebSocket with connection management
  - NAMING: WebSocket endpoints for job progress and system events
  - DEPENDENCIES: DashboardWebSocketManager
  - PLACEMENT: WebSocket endpoints in dashboard/backend/api/v1/endpoints/

Task 13: CREATE dashboard/backend/main.py
  - IMPLEMENT: FastAPI application factory with CORS and lifespan
  - FOLLOW pattern: src/core/app.py application creation
  - NAMING: create_dashboard_app async function, include routers
  - FEATURES: CORS middleware, context initialization, router inclusion
  - PLACEMENT: Main application in dashboard/backend/

Task 14: CREATE dashboard/frontend/src/dataProvider.js
  - IMPLEMENT: React Admin data provider for API integration
  - FOLLOW pattern: AI docs react_admin_dashboard_patterns.md data provider
  - NAMING: dataProvider object with getList, getOne, update, create, delete
  - FEATURES: Pagination, sorting, filtering, error handling
  - PLACEMENT: Frontend data layer in dashboard/frontend/src/

Task 15: CREATE dashboard/frontend/src/components/configuration/ConfigurationForm.js
  - IMPLEMENT: React Admin configuration management interface
  - FOLLOW pattern: AI docs react_admin_dashboard_patterns.md configuration form
  - NAMING: ConfigurationForm component with categorized accordions
  - FEATURES: Form validation, sensitive field masking, category organization
  - PLACEMENT: Frontend components in dashboard/frontend/src/components/

Task 16: CREATE dashboard/frontend/src/components/jobs/JobQueue.js
  - IMPLEMENT: Job queue interface with real-time updates
  - FOLLOW pattern: AI docs react_admin_dashboard_patterns.md job list
  - NAMING: JobQueue component with WebSocket integration
  - FEATURES: Real-time progress bars, job status updates, job details
  - PLACEMENT: Frontend components in dashboard/frontend/src/components/

Task 17: CREATE dashboard/frontend/src/hooks/useWebSocket.js
  - IMPLEMENT: WebSocket React hook for real-time communication
  - FOLLOW pattern: AI docs websocket_job_management_patterns.md React hook
  - NAMING: useWebSocket and useJobProgress hooks
  - FEATURES: Connection management, reconnection, message handling
  - PLACEMENT: Frontend hooks in dashboard/frontend/src/hooks/

Task 18: CREATE dashboard/frontend/src/App.js
  - IMPLEMENT: React Admin application setup
  - FOLLOW pattern: AI docs react_admin_dashboard_patterns.md App structure
  - NAMING: App component with Admin, Resource components
  - DEPENDENCIES: dataProvider, components, routing
  - PLACEMENT: Main frontend application in dashboard/frontend/src/

Task 19: CREATE tests/unit/dashboard/backend/test_config_service.py
  - IMPLEMENT: Unit tests for configuration service
  - FOLLOW pattern: tests/unit/services/test_rag_service.py test structure
  - NAMING: test_* functions with descriptive names
  - COVERAGE: All ConfigurationService methods, error cases
  - PLACEMENT: Backend tests in tests/unit/dashboard/backend/

Task 20: CREATE tests/unit/dashboard/backend/test_job_manager.py
  - IMPLEMENT: Unit tests for job management
  - FOLLOW pattern: tests/unit/services/ test patterns with mocking
  - NAMING: TestDashboardJobManager class with test methods
  - COVERAGE: Job execution, progress tracking, error handling
  - PLACEMENT: Backend tests in tests/unit/dashboard/backend/
```

### Implementation Patterns & Key Details

```python
# Configuration Service Pattern
class ConfigurationService:
    """Service for managing MCP server configuration"""
    
    def __init__(self, env_file_path: str = ".env"):
        self.env_file_path = env_file_path
        self.config_schema = self._load_config_schema()
    
    async def get_all_settings(self) -> Dict[str, ConfigSetting]:
        """Get all configuration settings organized by category"""
        # PATTERN: Read .env file and organize by categories
        # CRITICAL: Handle sensitive fields (API keys) with masking
        pass
    
    async def update_settings(self, changes: Dict[str, str]) -> ConfigUpdate:
        """Update configuration settings and write to .env file"""
        # PATTERN: Validate changes, backup existing .env, write new values
        # CRITICAL: Atomic write operation with rollback on failure
        pass

# Job Manager Pattern  
class DashboardJobManager:
    """Manage job execution with real-time progress tracking"""
    
    async def submit_job(self, request: JobRequest) -> str:
        """Submit job for execution with progress tracking"""
        # PATTERN: Create job ID, initialize progress tracker, start background task
        job_id = str(uuid.uuid4())
        tracker = JobProgressTracker(job_id)
        
        # Add progress callback for WebSocket updates
        async def progress_callback(update: ProgressUpdate):
            await self.websocket_manager.broadcast_job_progress(job_id, update.to_dict())
        
        tracker.add_progress_callback(progress_callback)
        # CRITICAL: Execute in background task with proper error handling
        return job_id

# WebSocket Connection Manager Pattern
class DashboardWebSocketManager:
    """Robust WebSocket connection management with health monitoring"""
    
    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        """Connect client with retry logic and health monitoring"""
        # PATTERN: Accept connection, store in pool, start ping monitoring
        # CRITICAL: Handle connection failures and cleanup gracefully
        pass
    
    async def broadcast_job_progress(self, job_id: str, progress_data: Dict):
        """Broadcast job progress to all subscribers"""
        # PATTERN: Find subscribers, send message, handle disconnections
        # CRITICAL: Clean up failed connections automatically
        pass

# React Admin Integration Pattern
const dataProvider = {
    getList: (resource, params) => {
        // PATTERN: Handle pagination, sorting, filtering
        // CRITICAL: Convert React Admin params to API format
        const { page, perPage } = params.pagination;
        const { field, order } = params.sort;
        const query = {
            sort: JSON.stringify([field, order]),
            range: JSON.stringify([(page - 1) * perPage, page * perPage - 1]),
            filter: JSON.stringify(params.filter),
        };
        const url = `${apiUrl}/${resource}?${new URLSearchParams(query)}`;
        return httpClient(url).then(({ headers, json }) => ({
            data: json.data,
            total: json.total,
        }));
    },
    // CRITICAL: Implement all CRUD operations for React Admin compatibility
};

# WebSocket Hook Pattern
export const useWebSocket = (clientId) => {
    // PATTERN: Connection management with automatic reconnection
    // CRITICAL: Proper cleanup on unmount, error handling, message routing
    const [socket, setSocket] = useState(null);
    const [connected, setConnected] = useState(false);
    
    const connect = useCallback(() => {
        const ws = new WebSocket(`${WEBSOCKET_URL}/ws/${clientId}`);
        // CRITICAL: Handle onopen, onmessage, onclose, onerror events
    }, [clientId]);
    
    useEffect(() => {
        connect();
        return () => disconnect(); // CRITICAL: Cleanup on unmount
    }, [connect, disconnect]);
};
```

### Integration Points

```yaml
DATABASE:
  - reuse: "src.clients.qdrant_client.QdrantClientWrapper"
  - pattern: "Singleton database connections with environment configuration"
  - critical: "Use same connection pool as MCP server"

SERVICES:
  - reuse: "src.services.rag_service.RagService"
  - pattern: "Service layer with dependency injection"
  - critical: "Import and reuse existing service classes"

MCP_TOOLS:
  - proxy: "src.tools.* modules"
  - pattern: "Tool execution with context mocking"
  - critical: "Proxy existing tool functions, don't reimplement"

CONFIGURATION:
  - share: ".env file and environment variables"
  - pattern: "Environment-based configuration"
  - critical: "Synchronize with existing MCP server configuration"

WEBSOCKET:
  - implement: "Real-time communication layer"
  - pattern: "Connection pooling with health monitoring"
  - critical: "Handle reconnection and error recovery"
```

## Validation Loop

### Level 1: Syntax & Style (Immediate Feedback)

```bash
# Run after each file creation - fix before proceeding
uv run ruff check dashboard/ --fix          # Auto-format and fix linting issues
uv run mypy dashboard/                      # Type checking
uv run ruff format dashboard/               # Ensure consistent formatting

# Project-wide validation
uv run ruff check . --fix
uv run mypy src/
uv run ruff format .

# Expected: Zero errors. If errors exist, READ output and fix before proceeding.
```

### Level 2: Unit Tests (Component Validation)

```bash
# Test each component as it's created
uv run pytest tests/unit/dashboard/backend/ -v
uv run pytest tests/unit/dashboard/frontend/ -v

# Full test suite for affected areas
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v

# Coverage validation
uv run pytest tests/unit/dashboard/ --cov=dashboard --cov-report=term-missing

# Expected: All tests pass. If failing, debug root cause and fix implementation.
```

### Level 3: Integration Testing (System Validation)

```bash
# Start all required services
docker-compose up -d                        # Start Qdrant, Neo4j, Redis

# Start MCP server (existing)
uv run -m src &                             # MCP server on port 8051
sleep 3

# Start dashboard backend
cd dashboard/backend && uvicorn main:app --host 0.0.0.0 --port 8080 &
sleep 3

# Start dashboard frontend  
cd dashboard/frontend && npm start &
sleep 5

# Health check validation
curl -f http://localhost:8080/api/v1/health || echo "Dashboard API health check failed"
curl -f http://localhost:3000 || echo "Frontend health check failed"

# WebSocket validation
echo '{"type": "ping"}' | websocat ws://localhost:8080/ws/test-client

# Database connectivity
curl -X GET http://localhost:8080/api/v1/metrics | jq .

# Configuration API validation
curl -X GET http://localhost:8080/api/v1/config | jq .

# Job execution validation
curl -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "get_available_sources", "parameters": {}}' \
  | jq .

# Expected: All integrations working, proper responses, no connection errors
```

### Level 4: End-to-End Dashboard Validation

```bash
# Full dashboard workflow testing

# Configuration Management Test
curl -X GET http://localhost:8080/api/v1/config/categories | jq .
curl -X PUT http://localhost:8080/api/v1/config/qdrant_host \
  -H "Content-Type: application/json" \
  -d '{"value": "localhost"}' | jq .

# Job Management Test  
JOB_ID=$(curl -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "crawl_single_page", "parameters": {"url": "https://example.com"}}' \
  | jq -r '.job_id')

# Monitor job progress
curl -X GET http://localhost:8080/api/v1/jobs/$JOB_ID | jq .

# WebSocket real-time updates test
websocat ws://localhost:8080/ws/test-client &
WS_PID=$!

# Submit job and monitor via WebSocket
curl -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "get_available_sources", "parameters": {}}' \
  | jq .

sleep 10
kill $WS_PID

# System metrics validation
curl -X GET http://localhost:8080/api/v1/metrics/system | jq .

# Frontend navigation test (requires Playwright or similar)
# npx playwright test dashboard-e2e.spec.js

# Expected: Full dashboard functionality working, real-time updates, job execution
```

## Final Validation Checklist

### Technical Validation

- [ ] All 4 validation levels completed successfully
- [ ] All tests pass: `uv run pytest tests/ -v`
- [ ] No linting errors: `uv run ruff check .`
- [ ] No type errors: `uv run mypy src/ dashboard/`
- [ ] No formatting issues: `uv run ruff format . --check`

### Feature Validation

- [ ] All 156+ configuration variables accessible via web interface
- [ ] All 12 MCP tools executable through dashboard with real-time progress
- [ ] WebSocket real-time updates working for job progress and system metrics
- [ ] Configuration export/import functionality working
- [ ] System health monitoring showing all services (Qdrant, Neo4j, Redis)
- [ ] Job history and results properly stored and displayed
- [ ] Error cases handled gracefully with proper error messages
- [ ] Mobile-responsive design working on different screen sizes

### Code Quality Validation

- [ ] Follows existing codebase patterns and naming conventions
- [ ] File placement matches desired codebase tree structure
- [ ] Anti-patterns avoided (no sync functions in async context, proper error handling)
- [ ] Dependencies properly managed and imported
- [ ] Configuration changes properly integrated
- [ ] Database connections reuse existing client patterns
- [ ] WebSocket connections properly managed with cleanup

### Documentation & Deployment

- [ ] Code is self-documenting with clear variable/function names
- [ ] Logs are informative but not verbose (ASCII only for Windows compatibility)
- [ ] Environment variables documented for new dashboard settings
- [ ] Integration points clearly defined and working
- [ ] Performance acceptable (dashboard response times < 2 seconds)

---

## Anti-Patterns to Avoid

- ❌ Don't modify existing MCP server code - create parallel service
- ❌ Don't skip validation because "it should work" - run all validation levels
- ❌ Don't ignore failing tests - fix them before proceeding
- ❌ Don't use sync functions in async context - maintain async patterns
- ❌ Don't hardcode values that should be config - use environment variables
- ❌ Don't catch all exceptions - be specific about error handling
- ❌ Don't use Unicode characters - ASCII only for Windows compatibility
- ❌ Don't recreate existing services - reuse src/services/* classes
- ❌ Don't implement new database clients - reuse src/clients/* patterns
- ❌ Don't bypass WebSocket connection management - use proper patterns