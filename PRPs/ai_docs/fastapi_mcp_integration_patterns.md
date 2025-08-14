# FastAPI + MCP Integration Patterns

This document provides critical patterns for integrating FastAPI web applications with existing MCP servers while maintaining proper architecture separation.

## Architecture Pattern: Parallel Services

**CRITICAL**: Do NOT modify the existing MCP server. Create a parallel FastAPI service that shares database access.

```
Existing MCP Server (Port 8051)     New Dashboard API (Port 8080)
├── FastMCP Framework               ├── FastAPI Framework  
├── MCP Tools (JSON-RPC)           ├── HTTP REST Endpoints
├── SSE Transport                  ├── WebSocket Support
└── Shared Databases ──────────────┴── Shared Databases
    ├── Qdrant (Port 6333)
    ├── Neo4j (Port 7687) 
    └── Redis (Port 6379)
```

## Database Sharing Pattern

### Reuse Existing Client Classes
```python
# dashboard/dependencies.py
from src.clients.qdrant_client import QdrantClientWrapper
from src.services.rag_service import RagService
from functools import lru_cache

@lru_cache()
def get_qdrant_client() -> QdrantClientWrapper:
    """Reuse existing Qdrant client wrapper"""
    return QdrantClientWrapper()

@lru_cache() 
def get_rag_service() -> RagService:
    """Reuse existing RAG service"""
    return RagService(get_qdrant_client())
```

### Context Management Pattern
```python
# dashboard/core/context.py
from src.core.context import Crawl4AIContext, ContextSingleton
from dataclasses import dataclass

@dataclass
class DashboardContext:
    """Dashboard-specific context that wraps MCP context"""
    mcp_context: Crawl4AIContext
    
    @classmethod
    async def create(cls):
        """Create dashboard context using existing MCP context"""
        singleton = ContextSingleton()
        mcp_context = await singleton.get_context(None)  # No server needed
        return cls(mcp_context=mcp_context)
    
    @property
    def qdrant_client(self):
        return self.mcp_context.qdrant_client
    
    @property
    def rag_service(self):
        return RagService(self.qdrant_client)
```

## Configuration Management Integration

### Environment Variable Sharing
```python
# dashboard/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class DashboardSettings(BaseSettings):
    # Dashboard-specific settings
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8080
    enable_cors: bool = True
    cors_origins: list = ["http://localhost:3000"]
    
    # Reuse MCP environment variables
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_password: str = "password123"
    
    # Feature flags (shared with MCP)
    use_hybrid_search: bool = False
    use_reranking: bool = False
    use_knowledge_graph: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

## Tool Execution Pattern

### MCP Tool Proxy Pattern
```python
# dashboard/services/mcp_proxy.py
import asyncio
import json
from typing import Dict, Any

class MCPToolProxy:
    """Proxy to execute MCP tools from HTTP API"""
    
    def __init__(self, dashboard_context: DashboardContext):
        self.context = dashboard_context
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MCP tool with given parameters"""
        
        # Map of tool names to actual implementations
        tool_mapping = {
            "crawl_single_page": self._crawl_single_page,
            "smart_crawl_github": self._smart_crawl_github,
            "perform_rag_query": self._perform_rag_query,
            # Add other tools as needed
        }
        
        if tool_name not in tool_mapping:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        try:
            result = await tool_mapping[tool_name](**parameters)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _crawl_single_page(self, url: str) -> str:
        """Execute crawl_single_page tool logic"""
        from src.tools.web_tools import crawl_single_page
        # Create mock context for tool execution
        mock_context = type('MockContext', (), {
            'request_context': type('RequestContext', (), {
                'lifespan_context': self.context.mcp_context
            })()
        })()
        return await crawl_single_page(mock_context, url)
```

## WebSocket Real-Time Updates

### Connection Manager with Job Tracking
```python
# dashboard/websocket/manager.py
from fastapi import WebSocket
from typing import Dict, List
import json
import asyncio

class DashboardWebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.job_subscribers: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = []
        self.active_connections[client_id].append(websocket)
    
    async def subscribe_to_job(self, websocket: WebSocket, job_id: str):
        """Subscribe websocket to job updates"""
        if job_id not in self.job_subscribers:
            self.job_subscribers[job_id] = []
        self.job_subscribers[job_id].append(websocket)
    
    async def notify_job_progress(self, job_id: str, progress_data: Dict):
        """Notify all subscribers of job progress"""
        if job_id in self.job_subscribers:
            message = json.dumps({
                "type": "job_progress",
                "job_id": job_id,
                "data": progress_data
            })
            
            disconnected = []
            for websocket in self.job_subscribers[job_id]:
                try:
                    await websocket.send_text(message)
                except:
                    disconnected.append(websocket)
            
            # Clean up disconnected websockets
            for ws in disconnected:
                self.job_subscribers[job_id].remove(ws)
```

## Background Job Management

### Job Queue Integration
```python
# dashboard/jobs/manager.py
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime
import uuid

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class DashboardJob:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class DashboardJobManager:
    def __init__(self, mcp_proxy: MCPToolProxy, websocket_manager: DashboardWebSocketManager):
        self.mcp_proxy = mcp_proxy
        self.websocket_manager = websocket_manager
        self.jobs: Dict[str, DashboardJob] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
    
    async def submit_job(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Submit a new job for execution"""
        job = DashboardJob(tool_name=tool_name, parameters=parameters)
        self.jobs[job.id] = job
        
        # Start job execution in background
        task = asyncio.create_task(self._execute_job(job))
        self.running_tasks[job.id] = task
        
        return job.id
    
    async def _execute_job(self, job: DashboardJob):
        """Execute job and update progress"""
        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            
            await self.websocket_manager.notify_job_progress(job.id, {
                "status": job.status.value,
                "progress": 0,
                "message": "Job started"
            })
            
            # Execute the actual tool
            result = await self.mcp_proxy.execute_tool(job.tool_name, job.parameters)
            
            if result["success"]:
                job.status = JobStatus.COMPLETED
                job.result = result["result"]
                job.progress = 100
            else:
                job.status = JobStatus.FAILED
                job.error = result["error"]
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
        
        finally:
            job.completed_at = datetime.now()
            await self.websocket_manager.notify_job_progress(job.id, {
                "status": job.status.value,
                "progress": job.progress,
                "result": job.result,
                "error": job.error
            })
            
            # Clean up task
            if job.id in self.running_tasks:
                del self.running_tasks[job.id]
```

## FastAPI Application Structure

### Main Application Factory
```python
# dashboard/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import DashboardSettings
from .api.v1.api import api_router
from .websocket.manager import DashboardWebSocketManager
from .jobs.manager import DashboardJobManager
from .services.mcp_proxy import MCPToolProxy
from .core.context import DashboardContext

async def create_dashboard_app() -> FastAPI:
    """Create FastAPI dashboard application"""
    settings = DashboardSettings()
    
    app = FastAPI(
        title="MCP Dashboard API",
        description="Configuration and monitoring dashboard for MCP server",
        version="1.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize dashboard context
    dashboard_context = await DashboardContext.create()
    
    # Initialize managers
    websocket_manager = DashboardWebSocketManager()
    mcp_proxy = MCPToolProxy(dashboard_context)
    job_manager = DashboardJobManager(mcp_proxy, websocket_manager)
    
    # Store in app state
    app.state.dashboard_context = dashboard_context
    app.state.websocket_manager = websocket_manager
    app.state.job_manager = job_manager
    
    # Include API routes
    app.include_router(api_router, prefix="/api/v1")
    
    return app

if __name__ == "__main__":
    import uvicorn
    app = asyncio.run(create_dashboard_app())
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

## Critical Integration Points

1. **DO NOT MODIFY** existing MCP server code
2. **REUSE** existing client classes and services
3. **SHARE** environment variables and configuration
4. **PROXY** MCP tool execution through HTTP API
5. **MAINTAIN** database connection sharing
6. **FOLLOW** existing error handling patterns
7. **RESPECT** existing logging and monitoring

This pattern ensures clean separation while maximizing code reuse and maintaining consistency with the existing MCP server architecture.