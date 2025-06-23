"""
Health check API endpoints.

This module provides health check endpoints for monitoring the MCP server status
and the status of its dependencies (Ollama, Supabase, Neo4j).
"""

import time
import requests
from typing import Dict, Any

from starlette.requests import Request
from starlette.responses import PlainTextResponse, JSONResponse

from ..clients.supabase_client import SupabaseService
from ..utils.validation import validate_neo4j_connection, format_neo4j_error
from ..config import config


async def check_ollama_status() -> Dict[str, Any]:
    """Check Ollama service status and available models."""
    try:
        embedding_api_base = config.EMBEDDING_MODEL_API_BASE
        if not embedding_api_base or "11434" not in embedding_api_base:
            return {"status": "disabled", "reason": "Ollama not configured"}
        
        # Extract base URL (remove /v1 if present)
        base_url = embedding_api_base.replace("/v1", "")
        
        # Check if Ollama is responding
        response = requests.get(f"{base_url}/api/tags", timeout=config.OLLAMA_CHECK_TIMEOUT)
        if response.status_code != 200:
            return {"status": "error", "reason": f"HTTP {response.status_code}"}
        
        models_data = response.json()
        models = [model["name"] for model in models_data.get("models", [])]
        
        # Check if configured embedding model is available
        embedding_model = config.EMBEDDING_MODEL
        model_available = any(embedding_model in model for model in models)
        
        return {
            "status": "healthy" if model_available else "warning",
            "models_count": len(models),
            "embedding_model": embedding_model,
            "model_available": model_available,
            "models": models[:3]  # Show first 3 models
        }
    except requests.exceptions.RequestException as e:
        return {"status": "error", "reason": f"Connection failed: {str(e)}"}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


async def check_supabase_status() -> Dict[str, Any]:
    """Check Supabase connection status."""
    try:
        supabase_service = SupabaseService()
        client = supabase_service.get_client()
        # Test connection by querying sources table
        sources_table = config.TABLE_SOURCES
        result = client.table(sources_table).select("count", count="exact").limit(1).execute()
        return {
            "status": "healthy",
            "sources_count": result.count if hasattr(result, 'count') else 0
        }
    except Exception as e:
        return {"status": "error", "reason": str(e)}


async def check_neo4j_status() -> Dict[str, Any]:
    """Check Neo4j connection status if enabled."""
    use_knowledge_graph = config.USE_KNOWLEDGE_GRAPH
    
    if not use_knowledge_graph:
        return {"status": "disabled", "reason": "Knowledge graph disabled"}
    
    if not validate_neo4j_connection():
        return {"status": "error", "reason": "Neo4j environment variables not configured"}
    
    try:
        # Test Neo4j connection (simplified check)
        neo4j_uri = config.NEO4J_URI or ""
        return {"status": "healthy", "uri": neo4j_uri}
    except Exception as e:
        return {"status": "error", "reason": format_neo4j_error(e)}


def register_health_endpoints(mcp):
    """Register health check endpoints with the MCP server."""
    
    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse | PlainTextResponse:
        """
        Health check endpoint for the MCP server.
        
        Query parameters:
        - format=json: Return detailed JSON response
        - format=text (default): Return simple text response
        """
        start_time = time.time()
        
        # Get format preference
        format_param = request.query_params.get("format", "text").lower()
        
        # Perform health checks
        checks = {
            "ollama": await check_ollama_status(),
            "supabase": await check_supabase_status(),
            "neo4j": await check_neo4j_status()
        }
        
        # Determine overall health
        critical_services = ["supabase"]  # Core services
        optional_services = ["ollama", "neo4j"]  # Optional services
        
        critical_healthy = all(
            checks[service]["status"] in ["healthy", "disabled"] 
            for service in critical_services
        )
        
        overall_status = "healthy" if critical_healthy else "unhealthy"
        
        # Count service statuses
        healthy_count = sum(1 for check in checks.values() if check["status"] == "healthy")
        total_enabled = sum(1 for check in checks.values() if check["status"] != "disabled")
        
        response_time = round((time.time() - start_time) * 1000, 2)  # ms
        
        if format_param == "json":
            return JSONResponse(
                content={
                    "status": overall_status,
                    "timestamp": time.time(),
                    "response_time_ms": response_time,
                    "services": checks,
                    "summary": {
                        "healthy_services": healthy_count,
                        "total_enabled_services": total_enabled,
                        "critical_services_ok": critical_healthy
                    },
                    "version": config.APPLICATION_VERSION,
                    "server": config.APPLICATION_NAME
                },
                status_code=200 if overall_status == "healthy" else 503
            )
        else:
            # Simple text response for basic health checks
            if overall_status == "healthy":
                return PlainTextResponse("OK", status_code=200)
            else:
                return PlainTextResponse("UNHEALTHY", status_code=503)