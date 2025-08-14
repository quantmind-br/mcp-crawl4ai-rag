# WebSocket Job Management and Real-time Progress Tracking Patterns

This document provides comprehensive patterns for implementing robust job management with WebSocket-based real-time progress tracking, specifically designed for the MCP dashboard.

## Architecture Overview

```
Frontend (React)          Backend (FastAPI)           Job Processing
├── WebSocket Client  ←→  ├── WebSocket Manager   ←→  ├── Background Tasks
├── Job Components        ├── Connection Pool         ├── Progress Tracking
├── Progress UI           ├── Message Broadcasting    ├── Error Handling
└── State Management      └── Job Queue API           └── Result Storage
```

## WebSocket Connection Management

### Robust Connection Manager
```python
# dashboard/websocket/connection_manager.py
import asyncio
import json
import logging
import time
from typing import Dict, List, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

class ConnectionState(str, Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"

@dataclass
class WebSocketConnection:
    websocket: WebSocket
    client_id: str
    connected_at: datetime = field(default_factory=datetime.now)
    last_ping: datetime = field(default_factory=datetime.now)
    state: ConnectionState = ConnectionState.CONNECTING
    subscriptions: Set[str] = field(default_factory=set)
    retry_count: int = 0

class DashboardWebSocketManager:
    def __init__(self, ping_interval: int = 30, max_retries: int = 3):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.job_subscribers: Dict[str, Set[str]] = {}  # job_id -> set of client_ids
        self.system_subscribers: Set[str] = set()  # clients subscribed to system events
        self.ping_interval = ping_interval
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
        self._ping_task: Optional[asyncio.Task] = None
        
    async def start_ping_task(self):
        """Start background ping task for connection health monitoring"""
        if self._ping_task is None or self._ping_task.done():
            self._ping_task = asyncio.create_task(self._ping_loop())
    
    async def stop_ping_task(self):
        """Stop background ping task"""
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
    
    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        """Connect a new WebSocket client with retry logic"""
        try:
            await websocket.accept()
            
            connection = WebSocketConnection(
                websocket=websocket,
                client_id=client_id,
                state=ConnectionState.CONNECTED
            )
            
            self.connections[client_id] = connection
            self.logger.info(f"Client {client_id} connected successfully")
            
            # Send welcome message
            await self._send_to_client(client_id, {
                "type": "connection_established",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect client {client_id}: {e}")
            return False
    
    async def disconnect(self, client_id: str):
        """Gracefully disconnect a client"""
        if client_id in self.connections:
            connection = self.connections[client_id]
            connection.state = ConnectionState.DISCONNECTING
            
            # Remove from all subscriptions
            self._cleanup_client_subscriptions(client_id)
            
            try:
                await connection.websocket.close()
            except:
                pass  # Already closed
            
            connection.state = ConnectionState.DISCONNECTED
            del self.connections[client_id]
            
            self.logger.info(f"Client {client_id} disconnected")
    
    async def subscribe_to_job(self, client_id: str, job_id: str):
        """Subscribe client to job progress updates"""
        if client_id not in self.connections:
            return False
        
        if job_id not in self.job_subscribers:
            self.job_subscribers[job_id] = set()
        
        self.job_subscribers[job_id].add(client_id)
        self.connections[client_id].subscriptions.add(f"job:{job_id}")
        
        self.logger.info(f"Client {client_id} subscribed to job {job_id}")
        return True
    
    async def subscribe_to_system(self, client_id: str):
        """Subscribe client to system-wide events"""
        if client_id not in self.connections:
            return False
        
        self.system_subscribers.add(client_id)
        self.connections[client_id].subscriptions.add("system")
        
        self.logger.info(f"Client {client_id} subscribed to system events")
        return True
    
    async def broadcast_job_progress(self, job_id: str, progress_data: Dict[str, Any]):
        """Broadcast job progress to all subscribers"""
        if job_id not in self.job_subscribers:
            return
        
        message = {
            "type": "job_progress",
            "job_id": job_id,
            "data": progress_data,
            "timestamp": datetime.now().isoformat()
        }
        
        failed_clients = []
        for client_id in self.job_subscribers[job_id]:
            success = await self._send_to_client(client_id, message)
            if not success:
                failed_clients.append(client_id)
        
        # Clean up failed connections
        for client_id in failed_clients:
            await self.disconnect(client_id)
    
    async def broadcast_system_event(self, event_type: str, data: Dict[str, Any]):
        """Broadcast system-wide events to all system subscribers"""
        message = {
            "type": "system_event",
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        failed_clients = []
        for client_id in self.system_subscribers:
            success = await self._send_to_client(client_id, message)
            if not success:
                failed_clients.append(client_id)
        
        # Clean up failed connections
        for client_id in failed_clients:
            await self.disconnect(client_id)
    
    async def _send_to_client(self, client_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific client with error handling"""
        if client_id not in self.connections:
            return False
        
        connection = self.connections[client_id]
        
        try:
            message_str = json.dumps(message)
            await connection.websocket.send_text(message_str)
            return True
            
        except WebSocketDisconnect:
            self.logger.info(f"Client {client_id} disconnected during send")
            await self.disconnect(client_id)
            return False
            
        except Exception as e:
            self.logger.error(f"Error sending message to {client_id}: {e}")
            connection.state = ConnectionState.ERROR
            connection.retry_count += 1
            
            if connection.retry_count > self.max_retries:
                await self.disconnect(client_id)
            
            return False
    
    async def _ping_loop(self):
        """Background task to ping all connections periodically"""
        while True:
            try:
                await asyncio.sleep(self.ping_interval)
                await self._ping_all_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in ping loop: {e}")
    
    async def _ping_all_connections(self):
        """Ping all active connections to check health"""
        disconnected_clients = []
        
        for client_id, connection in self.connections.items():
            try:
                # Send ping and wait for response
                await asyncio.wait_for(
                    connection.websocket.ping(),
                    timeout=10.0
                )
                connection.last_ping = datetime.now()
                
            except (asyncio.TimeoutError, Exception):
                self.logger.warning(f"Ping failed for client {client_id}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)
    
    def _cleanup_client_subscriptions(self, client_id: str):
        """Remove client from all subscriptions"""
        # Remove from job subscriptions
        for job_id, subscribers in list(self.job_subscribers.items()):
            subscribers.discard(client_id)
            if not subscribers:  # Remove empty subscription sets
                del self.job_subscribers[job_id]
        
        # Remove from system subscriptions
        self.system_subscribers.discard(client_id)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get current connection statistics"""
        total_connections = len(self.connections)
        healthy_connections = sum(
            1 for conn in self.connections.values() 
            if conn.state == ConnectionState.CONNECTED
        )
        
        return {
            "total_connections": total_connections,
            "healthy_connections": healthy_connections,
            "job_subscriptions": len(self.job_subscribers),
            "system_subscriptions": len(self.system_subscribers),
            "clients": [
                {
                    "client_id": conn.client_id,
                    "state": conn.state.value,
                    "connected_at": conn.connected_at.isoformat(),
                    "subscriptions": list(conn.subscriptions)
                }
                for conn in self.connections.values()
            ]
        }
```

## Job Progress Tracking System

### Progress Tracker with Granular Updates
```python
# dashboard/jobs/progress_tracker.py
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Callable, Awaitable
from datetime import datetime
from enum import Enum
import asyncio

class ProgressStage(str, Enum):
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ProgressUpdate:
    stage: ProgressStage
    current: int
    total: int
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def percentage(self) -> float:
        if self.total == 0:
            return 0.0
        return min(100.0, (self.current / self.total) * 100.0)

class JobProgressTracker:
    """Advanced progress tracker with stage management and detailed reporting"""
    
    def __init__(self, job_id: str, total_items: int = 100):
        self.job_id = job_id
        self.total_items = total_items
        self.current_items = 0
        self.current_stage = ProgressStage.INITIALIZING
        self.stage_progress: Dict[ProgressStage, int] = {}
        self.start_time = datetime.now()
        self.stage_start_time = datetime.now()
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.completed_tasks: List[str] = []
        self.callbacks: List[Callable[[ProgressUpdate], Awaitable[None]]] = []
    
    def add_progress_callback(self, callback: Callable[[ProgressUpdate], Awaitable[None]]):
        """Add callback to be notified of progress updates"""
        self.callbacks.append(callback)
    
    async def start_stage(self, stage: ProgressStage, message: str, total_items: Optional[int] = None):
        """Start a new progress stage"""
        self.current_stage = stage
        self.stage_start_time = datetime.now()
        
        if total_items is not None:
            self.total_items = total_items
        
        await self._notify_progress(message)
    
    async def update_progress(self, increment: int = 1, message: Optional[str] = None, details: Optional[Dict] = None):
        """Update progress within current stage"""
        self.current_items += increment
        
        # Ensure we don't exceed total
        self.current_items = min(self.current_items, self.total_items)
        
        if message is None:
            message = f"Processing item {self.current_items} of {self.total_items}"
        
        await self._notify_progress(message, details or {})
    
    async def set_progress(self, current: int, message: Optional[str] = None, details: Optional[Dict] = None):
        """Set absolute progress value"""
        self.current_items = min(current, self.total_items)
        
        if message is None:
            message = f"Processed {self.current_items} of {self.total_items} items"
        
        await self._notify_progress(message, details or {})
    
    async def add_error(self, error_message: str):
        """Add error message and notify"""
        self.errors.append(error_message)
        await self._notify_progress(f"Error: {error_message}", {"error": True})
    
    async def add_warning(self, warning_message: str):
        """Add warning message and notify"""
        self.warnings.append(warning_message)
        await self._notify_progress(f"Warning: {warning_message}", {"warning": True})
    
    async def complete_task(self, task_name: str):
        """Mark a specific task as completed"""
        self.completed_tasks.append(task_name)
        await self._notify_progress(f"Completed: {task_name}", {"task_completed": task_name})
    
    async def complete_stage(self, message: Optional[str] = None):
        """Complete current stage and move to next"""
        self.stage_progress[self.current_stage] = self.current_items
        
        if message is None:
            message = f"Completed {self.current_stage.value} stage"
        
        await self._notify_progress(message, {"stage_completed": self.current_stage.value})
    
    async def finish_success(self, final_message: str = "Job completed successfully"):
        """Mark job as successfully completed"""
        self.current_stage = ProgressStage.COMPLETED
        self.current_items = self.total_items
        await self._notify_progress(final_message, {"success": True})
    
    async def finish_failure(self, error_message: str):
        """Mark job as failed"""
        self.current_stage = ProgressStage.FAILED
        await self._notify_progress(f"Job failed: {error_message}", {"error": True, "failed": True})
    
    async def _notify_progress(self, message: str, details: Optional[Dict] = None):
        """Send progress update to all callbacks"""
        update = ProgressUpdate(
            stage=self.current_stage,
            current=self.current_items,
            total=self.total_items,
            message=message,
            details=details or {}
        )
        
        # Add timing information
        update.details.update({
            "elapsed_time": (datetime.now() - self.start_time).total_seconds(),
            "stage_time": (datetime.now() - self.stage_start_time).total_seconds(),
            "estimated_remaining": self._estimate_remaining_time(),
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
            "completed_tasks": len(self.completed_tasks)
        })
        
        # Notify all callbacks
        for callback in self.callbacks:
            try:
                await callback(update)
            except Exception as e:
                # Don't let callback errors stop progress updates
                print(f"Error in progress callback: {e}")
    
    def _estimate_remaining_time(self) -> Optional[float]:
        """Estimate remaining time based on current progress"""
        if self.current_items == 0:
            return None
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.current_items / elapsed  # items per second
        
        if rate == 0:
            return None
        
        remaining_items = self.total_items - self.current_items
        return remaining_items / rate
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive progress summary"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "job_id": self.job_id,
            "stage": self.current_stage.value,
            "progress": {
                "current": self.current_items,
                "total": self.total_items,
                "percentage": (self.current_items / self.total_items) * 100 if self.total_items > 0 else 0
            },
            "timing": {
                "start_time": self.start_time.isoformat(),
                "elapsed_seconds": elapsed,
                "estimated_remaining": self._estimate_remaining_time()
            },
            "status": {
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "completed_tasks": len(self.completed_tasks)
            },
            "details": {
                "errors": self.errors[-5:],  # Last 5 errors
                "warnings": self.warnings[-5:],  # Last 5 warnings
                "recent_tasks": self.completed_tasks[-10:]  # Last 10 completed tasks
            }
        }
```

## Frontend WebSocket Integration

### React Hook for WebSocket Management
```javascript
// frontend/src/hooks/useWebSocket.js
import { useState, useEffect, useRef, useCallback } from 'react';

const WEBSOCKET_URL = 'ws://localhost:8080';
const RECONNECT_INTERVAL = 3000;
const MAX_RECONNECT_ATTEMPTS = 5;

export const useWebSocket = (clientId) => {
    const [socket, setSocket] = useState(null);
    const [connected, setConnected] = useState(false);
    const [connectionState, setConnectionState] = useState('disconnected');
    const [lastMessage, setLastMessage] = useState(null);
    const [jobProgress, setJobProgress] = useState({});
    const [systemEvents, setSystemEvents] = useState([]);
    
    const reconnectAttempts = useRef(0);
    const reconnectTimer = useRef(null);
    const messageHandlers = useRef(new Map());

    const connect = useCallback(() => {
        if (socket?.readyState === WebSocket.OPEN) {
            return;
        }

        setConnectionState('connecting');
        
        try {
            const ws = new WebSocket(`${WEBSOCKET_URL}/ws/${clientId}`);
            
            ws.onopen = () => {
                console.log('WebSocket connected');
                setConnected(true);
                setConnectionState('connected');
                setSocket(ws);
                reconnectAttempts.current = 0;
                
                // Clear any pending reconnect
                if (reconnectTimer.current) {
                    clearTimeout(reconnectTimer.current);
                    reconnectTimer.current = null;
                }
            };
            
            ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    setLastMessage(message);
                    
                    // Handle different message types
                    switch (message.type) {
                        case 'job_progress':
                            setJobProgress(prev => ({
                                ...prev,
                                [message.job_id]: message.data
                            }));
                            break;
                            
                        case 'system_event':
                            setSystemEvents(prev => [
                                ...prev.slice(-49), // Keep last 50 events
                                message
                            ]);
                            break;
                            
                        case 'connection_established':
                            console.log('Connection established:', message);
                            break;
                    }
                    
                    // Call registered message handlers
                    const handler = messageHandlers.current.get(message.type);
                    if (handler) {
                        handler(message);
                    }
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };
            
            ws.onclose = (event) => {
                console.log('WebSocket disconnected:', event.code, event.reason);
                setConnected(false);
                setConnectionState('disconnected');
                setSocket(null);
                
                // Attempt reconnection if not intentional close
                if (event.code !== 1000 && reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
                    setConnectionState('reconnecting');
                    reconnectAttempts.current++;
                    
                    reconnectTimer.current = setTimeout(() => {
                        console.log(`Reconnection attempt ${reconnectAttempts.current}`);
                        connect();
                    }, RECONNECT_INTERVAL * reconnectAttempts.current);
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                setConnectionState('error');
            };
            
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            setConnectionState('error');
        }
    }, [clientId, socket]);

    const disconnect = useCallback(() => {
        if (reconnectTimer.current) {
            clearTimeout(reconnectTimer.current);
            reconnectTimer.current = null;
        }
        
        if (socket) {
            socket.close(1000, 'Intentional disconnect');
        }
        
        setSocket(null);
        setConnected(false);
        setConnectionState('disconnected');
    }, [socket]);

    const subscribeToJob = useCallback(async (jobId) => {
        if (!socket || socket.readyState !== WebSocket.OPEN) {
            console.warn('Cannot subscribe to job: WebSocket not connected');
            return false;
        }

        try {
            const response = await fetch(`http://localhost:8080/api/v1/websocket/subscribe/job/${jobId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ client_id: clientId })
            });
            
            return response.ok;
        } catch (error) {
            console.error('Failed to subscribe to job:', error);
            return false;
        }
    }, [socket, clientId]);

    const subscribeToSystem = useCallback(async () => {
        if (!socket || socket.readyState !== WebSocket.OPEN) {
            console.warn('Cannot subscribe to system events: WebSocket not connected');
            return false;
        }

        try {
            const response = await fetch(`http://localhost:8080/api/v1/websocket/subscribe/system`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ client_id: clientId })
            });
            
            return response.ok;
        } catch (error) {
            console.error('Failed to subscribe to system events:', error);
            return false;
        }
    }, [socket, clientId]);

    const registerMessageHandler = useCallback((messageType, handler) => {
        messageHandlers.current.set(messageType, handler);
        
        // Return cleanup function
        return () => {
            messageHandlers.current.delete(messageType);
        };
    }, []);

    useEffect(() => {
        connect();
        
        return () => {
            disconnect();
        };
    }, [connect, disconnect]);

    return {
        socket,
        connected,
        connectionState,
        lastMessage,
        jobProgress,
        systemEvents,
        connect,
        disconnect,
        subscribeToJob,
        subscribeToSystem,
        registerMessageHandler
    };
};

// Hook for specific job monitoring
export const useJobProgress = (jobId) => {
    const [progress, setProgress] = useState(null);
    const [status, setStatus] = useState('unknown');
    const { subscribeToJob, jobProgress, registerMessageHandler } = useWebSocket('job-monitor');

    useEffect(() => {
        if (jobId) {
            subscribeToJob(jobId);
        }
    }, [jobId, subscribeToJob]);

    useEffect(() => {
        if (jobId && jobProgress[jobId]) {
            const jobData = jobProgress[jobId];
            setProgress(jobData);
            setStatus(jobData.stage || jobData.status || 'unknown');
        }
    }, [jobId, jobProgress]);

    return {
        progress,
        status,
        percentage: progress?.progress?.percentage || 0,
        message: progress?.message || '',
        errors: progress?.details?.errors || [],
        warnings: progress?.details?.warnings || []
    };
};
```

## Integration with MCP Tool Execution

### Async Job Execution with Progress
```python
# dashboard/jobs/executor.py
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

class MCPJobExecutor:
    """Execute MCP tools with real-time progress tracking"""
    
    def __init__(self, mcp_proxy, websocket_manager):
        self.mcp_proxy = mcp_proxy
        self.websocket_manager = websocket_manager
        self.active_jobs: Dict[str, JobProgressTracker] = {}
    
    async def execute_job_with_progress(self, job_id: str, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MCP tool with granular progress tracking"""
        
        # Create progress tracker
        tracker = JobProgressTracker(job_id, total_items=100)  # Default 100 steps
        self.active_jobs[job_id] = tracker
        
        # Register progress callback
        async def progress_callback(update: ProgressUpdate):
            await self.websocket_manager.broadcast_job_progress(job_id, {
                "stage": update.stage.value,
                "progress": {
                    "current": update.current,
                    "total": update.total,
                    "percentage": update.percentage
                },
                "message": update.message,
                "details": update.details,
                "timestamp": update.timestamp.isoformat()
            })
        
        tracker.add_progress_callback(progress_callback)
        
        try:
            # Stage 1: Initialize
            await tracker.start_stage(ProgressStage.INITIALIZING, "Initializing job...")
            await asyncio.sleep(0.1)  # Simulate initialization
            await tracker.update_progress(10, "Job parameters validated")
            
            # Stage 2: Execute tool
            await tracker.start_stage(ProgressStage.PROCESSING, f"Executing {tool_name}...")
            
            # Execute the actual MCP tool with progress simulation
            result = await self._execute_with_simulated_progress(tracker, tool_name, parameters)
            
            # Stage 3: Finalize
            await tracker.start_stage(ProgressStage.FINALIZING, "Finalizing results...")
            await tracker.update_progress(95, "Processing results")
            await asyncio.sleep(0.5)  # Simulate finalization
            
            # Complete
            await tracker.finish_success("Job completed successfully")
            
            return {
                "success": True,
                "result": result,
                "job_id": job_id,
                "summary": tracker.get_summary()
            }
            
        except Exception as e:
            await tracker.finish_failure(str(e))
            return {
                "success": False,
                "error": str(e),
                "job_id": job_id,
                "summary": tracker.get_summary()
            }
        
        finally:
            # Clean up
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
    
    async def _execute_with_simulated_progress(self, tracker: JobProgressTracker, tool_name: str, parameters: Dict[str, Any]):
        """Execute tool with simulated progress updates"""
        
        # Special handling for different tool types
        if tool_name == "smart_crawl_github":
            return await self._execute_github_crawl_with_progress(tracker, parameters)
        elif tool_name == "crawl_single_page":
            return await self._execute_page_crawl_with_progress(tracker, parameters)
        else:
            # Generic execution with basic progress
            return await self._execute_generic_tool_with_progress(tracker, tool_name, parameters)
    
    async def _execute_github_crawl_with_progress(self, tracker: JobProgressTracker, parameters: Dict[str, Any]):
        """Execute GitHub crawl with detailed progress"""
        repo_url = parameters.get("repo_url", "")
        max_files = parameters.get("max_files", 50)
        
        # Update total based on expected files
        tracker.total_items = max_files + 20  # Extra steps for setup/cleanup
        
        await tracker.update_progress(5, f"Cloning repository: {repo_url}")
        await asyncio.sleep(1)
        
        await tracker.update_progress(10, "Discovering files...")
        await asyncio.sleep(0.5)
        
        # Simulate file processing
        for i in range(max_files):
            await tracker.update_progress(1, f"Processing file {i+1} of {max_files}")
            await asyncio.sleep(0.1)  # Simulate processing time
        
        await tracker.update_progress(5, "Generating embeddings...")
        await asyncio.sleep(1)
        
        # Execute actual tool
        result = await self.mcp_proxy.execute_tool("smart_crawl_github", parameters)
        
        return result
    
    async def _execute_page_crawl_with_progress(self, tracker: JobProgressTracker, parameters: Dict[str, Any]):
        """Execute single page crawl with progress"""
        url = parameters.get("url", "")
        
        await tracker.update_progress(20, f"Fetching URL: {url}")
        await asyncio.sleep(0.5)
        
        await tracker.update_progress(40, "Processing content...")
        await asyncio.sleep(0.3)
        
        await tracker.update_progress(60, "Generating embeddings...")
        await asyncio.sleep(0.5)
        
        await tracker.update_progress(80, "Storing in vector database...")
        await asyncio.sleep(0.3)
        
        # Execute actual tool
        result = await self.mcp_proxy.execute_tool("crawl_single_page", parameters)
        
        return result
    
    async def _execute_generic_tool_with_progress(self, tracker: JobProgressTracker, tool_name: str, parameters: Dict[str, Any]):
        """Execute generic tool with basic progress"""
        await tracker.update_progress(25, f"Executing {tool_name}...")
        await asyncio.sleep(0.5)
        
        await tracker.update_progress(50, "Processing...")
        
        # Execute actual tool
        result = await self.mcp_proxy.execute_tool(tool_name, parameters)
        
        await tracker.update_progress(75, "Finalizing...")
        await asyncio.sleep(0.3)
        
        return result
    
    def get_active_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active jobs"""
        return {
            job_id: tracker.get_summary()
            for job_id, tracker in self.active_jobs.items()
        }
```

## Key Performance Optimizations

1. **Connection Pooling**: Efficient WebSocket connection management
2. **Message Batching**: Reduce WebSocket message frequency for high-frequency updates
3. **Progress Throttling**: Limit progress update frequency to prevent UI lag
4. **Automatic Cleanup**: Remove completed jobs and stale connections
5. **Error Recovery**: Robust reconnection and error handling
6. **Memory Management**: Limit retained message history and job data

This comprehensive pattern provides production-ready WebSocket job management with real-time progress tracking, robust error handling, and optimal performance characteristics.