# React Admin Dashboard Implementation Patterns

This document provides specific patterns and implementation guidance for building the MCP dashboard frontend using React Admin framework.

## Architecture Overview

```
React Admin Dashboard (Port 3000)
├── Admin Layout (Material-UI)
├── Data Provider (Custom API integration)
├── Auth Provider (JWT/Token based)
├── Resource Components (Configuration, Jobs, Monitoring)
├── Custom Dashboard (Real-time metrics)
└── WebSocket Integration (Real-time updates)
```

## Core Technology Stack

```json
{
  "react-admin": "^4.16.0",
  "react": "^18.2.0",
  "material-ui": "^5.0.0",
  "react-hook-form": "^7.45.0",
  "recharts": "^2.8.0",
  "socket.io-client": "^4.7.0"
}
```

## Data Provider Pattern

### Custom API Data Provider
```javascript
// src/dataProvider.js
import { fetchUtils } from 'react-admin';

const httpClient = fetchUtils.fetchJson;
const apiUrl = 'http://localhost:8080/api/v1';

export const dataProvider = {
    getList: (resource, params) => {
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

    getOne: (resource, params) =>
        httpClient(`${apiUrl}/${resource}/${params.id}`).then(({ json }) => ({
            data: json,
        })),

    getMany: (resource, params) => {
        const query = {
            filter: JSON.stringify({ id: params.ids }),
        };
        const url = `${apiUrl}/${resource}?${new URLSearchParams(query)}`;
        return httpClient(url).then(({ json }) => ({ data: json }));
    },

    getManyReference: (resource, params) => {
        const { page, perPage } = params.pagination;
        const { field, order } = params.sort;
        const query = {
            sort: JSON.stringify([field, order]),
            range: JSON.stringify([(page - 1) * perPage, page * perPage - 1]),
            filter: JSON.stringify({
                ...params.filter,
                [params.target]: params.id,
            }),
        };
        const url = `${apiUrl}/${resource}?${new URLSearchParams(query)}`;

        return httpClient(url).then(({ headers, json }) => ({
            data: json,
            total: parseInt(headers.get('content-range').split('/').pop(), 10),
        }));
    },

    update: (resource, params) =>
        httpClient(`${apiUrl}/${resource}/${params.id}`, {
            method: 'PUT',
            body: JSON.stringify(params.data),
        }).then(({ json }) => ({ data: json })),

    updateMany: (resource, params) => {
        const query = {
            filter: JSON.stringify({ id: params.ids}),
        };
        return httpClient(`${apiUrl}/${resource}?${new URLSearchParams(query)}`, {
            method: 'PUT',
            body: JSON.stringify(params.data),
        }).then(({ json }) => ({ data: json }));
    },

    create: (resource, params) =>
        httpClient(`${apiUrl}/${resource}`, {
            method: 'POST',
            body: JSON.stringify(params.data),
        }).then(({ json }) => ({
            data: { ...params.data, id: json.id },
        })),

    delete: (resource, params) =>
        httpClient(`${apiUrl}/${resource}/${params.id}`, {
            method: 'DELETE',
        }).then(({ json }) => ({ data: json })),

    deleteMany: (resource, params) => {
        const query = {
            filter: JSON.stringify({ id: params.ids}),
        };
        return httpClient(`${apiUrl}/${resource}?${new URLSearchParams(query)}`, {
            method: 'DELETE',
        }).then(({ json }) => ({ data: json }));
    },
};
```

## Configuration Management Interface

### Configuration Form Component
```javascript
// src/components/configuration/ConfigurationForm.js
import React from 'react';
import {
    Form,
    TextInput,
    NumberInput,
    BooleanInput,
    SelectInput,
    Card,
    CardContent,
    Typography,
    Box,
    Accordion,
    AccordionSummary,
    AccordionDetails,
} from 'react-admin';
import { ExpandMore } from '@mui/icons-material';

const ConfigurationForm = () => {
    return (
        <Form>
            <Box p={2}>
                <Typography variant="h5" gutterBottom>
                    MCP Server Configuration
                </Typography>
                
                {/* Server Settings */}
                <Accordion>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                        <Typography variant="h6">Server Settings</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                        <Box display="flex" flexDirection="column" gap={2}>
                            <TextInput 
                                source="transport" 
                                label="Transport Protocol"
                                choices={[
                                    { id: 'sse', name: 'Server-Sent Events' },
                                    { id: 'stdio', name: 'Standard I/O' }
                                ]}
                            />
                            <TextInput source="host" label="Host" defaultValue="0.0.0.0" />
                            <NumberInput source="port" label="Port" defaultValue={8051} />
                        </Box>
                    </AccordionDetails>
                </Accordion>

                {/* AI Models */}
                <Accordion>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                        <Typography variant="h6">AI Models</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                        <Box display="flex" flexDirection="column" gap={2}>
                            <SelectInput
                                source="chat_model"
                                label="Chat Model"
                                choices={[
                                    { id: 'gpt-4o-mini', name: 'GPT-4O Mini' },
                                    { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo' },
                                    { id: 'claude-3-haiku', name: 'Claude 3 Haiku' }
                                ]}
                            />
                            <TextInput source="chat_api_key" label="Chat API Key" type="password" />
                            <SelectInput
                                source="embeddings_model"
                                label="Embeddings Model"
                                choices={[
                                    { id: 'text-embedding-3-small', name: 'OpenAI Text Embedding 3 Small' },
                                    { id: 'Qwen/Qwen3-Embedding-0.6B', name: 'Qwen3 Embedding 0.6B' },
                                    { id: 'BAAI/bge-large-en-v1.5', name: 'BGE Large EN v1.5' }
                                ]}
                            />
                            <TextInput source="embeddings_api_key" label="Embeddings API Key" type="password" />
                        </Box>
                    </AccordionDetails>
                </Accordion>

                {/* RAG Features */}
                <Accordion>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                        <Typography variant="h6">RAG Features</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                        <Box display="flex" flexDirection="column" gap={2}>
                            <BooleanInput 
                                source="use_contextual_embeddings" 
                                label="Use Contextual Embeddings"
                                helperText="Enhance chunks with document context (+30% token usage, +15-25% accuracy)"
                            />
                            <BooleanInput 
                                source="use_hybrid_search" 
                                label="Use Hybrid Search"
                                helperText="Combine semantic + keyword search (+20-40% accuracy)"
                            />
                            <BooleanInput 
                                source="use_agentic_rag" 
                                label="Use Agentic RAG"
                                helperText="Extract and index code examples separately"
                            />
                            <BooleanInput 
                                source="use_reranking" 
                                label="Use Reranking"
                                helperText="Re-order search results (+50-100ms, +10-15% accuracy)"
                            />
                            <BooleanInput 
                                source="use_knowledge_graph" 
                                label="Use Knowledge Graph"
                                helperText="Neo4j integration for AI hallucination detection"
                            />
                        </Box>
                    </AccordionDetails>
                </Accordion>

                {/* Database Configuration */}
                <Accordion>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                        <Typography variant="h6">Database Configuration</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                        <Box display="flex" flexDirection="column" gap={2}>
                            <TextInput source="qdrant_host" label="Qdrant Host" defaultValue="localhost" />
                            <NumberInput source="qdrant_port" label="Qdrant Port" defaultValue={6333} />
                            <TextInput source="neo4j_uri" label="Neo4j URI" defaultValue="bolt://localhost:7687" />
                            <TextInput source="neo4j_user" label="Neo4j User" defaultValue="neo4j" />
                            <TextInput source="neo4j_password" label="Neo4j Password" type="password" />
                        </Box>
                    </AccordionDetails>
                </Accordion>
            </Box>
        </Form>
    );
};

export default ConfigurationForm;
```

## Job Management Interface

### Job Queue Component with Real-time Updates
```javascript
// src/components/jobs/JobQueue.js
import React, { useState, useEffect } from 'react';
import {
    List,
    Datagrid,
    TextField,
    DateField,
    ChipField,
    FunctionField,
    Card,
    CardContent,
    Typography,
    LinearProgress,
    Box,
    IconButton,
    Dialog,
    DialogTitle,
    DialogContent,
} from 'react-admin';
import { PlayArrow, Pause, Stop, Visibility } from '@mui/icons-material';
import io from 'socket.io-client';

const JobStatusChip = ({ record }) => {
    const statusColors = {
        pending: 'default',
        running: 'primary',
        completed: 'success',
        failed: 'error'
    };
    
    return (
        <ChipField 
            source="status" 
            color={statusColors[record.status]} 
            variant="outlined"
        />
    );
};

const JobProgressBar = ({ record }) => {
    if (record.status === 'pending') {
        return <Typography variant="body2">Waiting...</Typography>;
    }
    
    if (record.status === 'running') {
        return (
            <Box display="flex" alignItems="center" gap={1}>
                <LinearProgress 
                    variant="determinate" 
                    value={record.progress || 0} 
                    sx={{ width: 100 }}
                />
                <Typography variant="body2">{record.progress || 0}%</Typography>
            </Box>
        );
    }
    
    return (
        <Typography variant="body2">
            {record.status === 'completed' ? 'Done' : 'Failed'}
        </Typography>
    );
};

const JobList = () => {
    const [jobs, setJobs] = useState([]);
    const [selectedJob, setSelectedJob] = useState(null);
    const [dialogOpen, setDialogOpen] = useState(false);

    useEffect(() => {
        // Connect to WebSocket for real-time updates
        const socket = io('ws://localhost:8080');
        
        socket.on('job_progress', (data) => {
            setJobs(prevJobs => 
                prevJobs.map(job => 
                    job.id === data.job_id 
                        ? { ...job, ...data.data }
                        : job
                )
            );
        });

        socket.on('job_created', (data) => {
            setJobs(prevJobs => [...prevJobs, data.job]);
        });

        return () => socket.disconnect();
    }, []);

    const handleViewJob = (job) => {
        setSelectedJob(job);
        setDialogOpen(true);
    };

    return (
        <Card>
            <CardContent>
                <Typography variant="h6" gutterBottom>
                    Job Queue
                </Typography>
                <List>
                    <Datagrid>
                        <TextField source="id" label="Job ID" />
                        <TextField source="tool_name" label="Tool" />
                        <JobStatusChip label="Status" />
                        <FunctionField 
                            label="Progress" 
                            render={record => <JobProgressBar record={record} />}
                        />
                        <DateField source="created_at" label="Created" showTime />
                        <FunctionField
                            label="Actions"
                            render={record => (
                                <Box display="flex" gap={1}>
                                    <IconButton 
                                        size="small" 
                                        onClick={() => handleViewJob(record)}
                                    >
                                        <Visibility />
                                    </IconButton>
                                    {record.status === 'running' && (
                                        <IconButton size="small" color="error">
                                            <Stop />
                                        </IconButton>
                                    )}
                                </Box>
                            )}
                        />
                    </Datagrid>
                </List>

                <Dialog 
                    open={dialogOpen} 
                    onClose={() => setDialogOpen(false)}
                    maxWidth="md"
                    fullWidth
                >
                    <DialogTitle>Job Details</DialogTitle>
                    <DialogContent>
                        {selectedJob && (
                            <Box display="flex" flexDirection="column" gap={2}>
                                <Typography><strong>ID:</strong> {selectedJob.id}</Typography>
                                <Typography><strong>Tool:</strong> {selectedJob.tool_name}</Typography>
                                <Typography><strong>Status:</strong> {selectedJob.status}</Typography>
                                <Typography><strong>Progress:</strong> {selectedJob.progress}%</Typography>
                                {selectedJob.result && (
                                    <Box>
                                        <Typography variant="h6">Result:</Typography>
                                        <pre style={{ whiteSpace: 'pre-wrap', fontSize: '12px' }}>
                                            {JSON.stringify(selectedJob.result, null, 2)}
                                        </pre>
                                    </Box>
                                )}
                                {selectedJob.error && (
                                    <Box>
                                        <Typography variant="h6" color="error">Error:</Typography>
                                        <Typography color="error">{selectedJob.error}</Typography>
                                    </Box>
                                )}
                            </Box>
                        )}
                    </DialogContent>
                </Dialog>
            </CardContent>
        </Card>
    );
};

export default JobList;
```

## Real-time Monitoring Dashboard

### System Metrics Dashboard
```javascript
// src/components/dashboard/SystemMetrics.js
import React, { useState, useEffect } from 'react';
import {
    Card,
    CardContent,
    Typography,
    Grid,
    Box,
    Chip,
} from 'react-admin';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    PieChart,
    Pie,
    Cell,
} from 'recharts';

const MetricCard = ({ title, value, unit, status }) => {
    const statusColors = {
        healthy: 'success',
        warning: 'warning',
        error: 'error'
    };

    return (
        <Card>
            <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                    <Typography variant="h6">{title}</Typography>
                    <Chip 
                        label={status} 
                        color={statusColors[status]} 
                        variant="outlined"
                        size="small"
                    />
                </Box>
                <Typography variant="h4" color="primary">
                    {value} {unit}
                </Typography>
            </CardContent>
        </Card>
    );
};

const SystemMetrics = () => {
    const [metrics, setMetrics] = useState({
        cpu_usage: 0,
        memory_usage: 0,
        active_jobs: 0,
        total_sources: 0,
        services: {
            qdrant: 'healthy',
            neo4j: 'healthy',
            redis: 'healthy'
        }
    });
    const [performanceData, setPerformanceData] = useState([]);

    useEffect(() => {
        // Fetch initial metrics
        const fetchMetrics = async () => {
            try {
                const response = await fetch('http://localhost:8080/api/v1/metrics');
                const data = await response.json();
                setMetrics(data);
                
                // Add to performance history
                setPerformanceData(prev => [
                    ...prev.slice(-19), // Keep last 20 points
                    {
                        timestamp: new Date().toLocaleTimeString(),
                        cpu: data.cpu_usage,
                        memory: data.memory_usage
                    }
                ]);
            } catch (error) {
                console.error('Failed to fetch metrics:', error);
            }
        };

        fetchMetrics();
        const interval = setInterval(fetchMetrics, 5000); // Update every 5 seconds

        return () => clearInterval(interval);
    }, []);

    return (
        <Grid container spacing={3}>
            <Grid item xs={12} md={3}>
                <MetricCard 
                    title="CPU Usage" 
                    value={metrics.cpu_usage} 
                    unit="%" 
                    status={metrics.cpu_usage > 80 ? 'error' : metrics.cpu_usage > 60 ? 'warning' : 'healthy'}
                />
            </Grid>
            <Grid item xs={12} md={3}>
                <MetricCard 
                    title="Memory Usage" 
                    value={metrics.memory_usage} 
                    unit="%" 
                    status={metrics.memory_usage > 85 ? 'error' : metrics.memory_usage > 70 ? 'warning' : 'healthy'}
                />
            </Grid>
            <Grid item xs={12} md={3}>
                <MetricCard 
                    title="Active Jobs" 
                    value={metrics.active_jobs} 
                    unit="" 
                    status={metrics.active_jobs > 10 ? 'warning' : 'healthy'}
                />
            </Grid>
            <Grid item xs={12} md={3}>
                <MetricCard 
                    title="Total Sources" 
                    value={metrics.total_sources} 
                    unit="" 
                    status="healthy"
                />
            </Grid>

            {/* Performance Chart */}
            <Grid item xs={12} md={8}>
                <Card>
                    <CardContent>
                        <Typography variant="h6" gutterBottom>
                            Performance Trends
                        </Typography>
                        <ResponsiveContainer width="100%" height={300}>
                            <LineChart data={performanceData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="timestamp" />
                                <YAxis />
                                <Tooltip />
                                <Line type="monotone" dataKey="cpu" stroke="#8884d8" name="CPU %" />
                                <Line type="monotone" dataKey="memory" stroke="#82ca9d" name="Memory %" />
                            </LineChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>
            </Grid>

            {/* Service Status */}
            <Grid item xs={12} md={4}>
                <Card>
                    <CardContent>
                        <Typography variant="h6" gutterBottom>
                            Service Status
                        </Typography>
                        <Box display="flex" flexDirection="column" gap={1}>
                            {Object.entries(metrics.services).map(([service, status]) => (
                                <Box key={service} display="flex" justifyContent="space-between" alignItems="center">
                                    <Typography variant="body1">{service.toUpperCase()}</Typography>
                                    <Chip 
                                        label={status} 
                                        color={status === 'healthy' ? 'success' : 'error'} 
                                        variant="outlined"
                                        size="small"
                                    />
                                </Box>
                            ))}
                        </Box>
                    </CardContent>
                </Card>
            </Grid>
        </Grid>
    );
};

export default SystemMetrics;
```

## Main Application Structure

### App.js with React Admin Setup
```javascript
// src/App.js
import React from 'react';
import { Admin, Resource, CustomRoutes } from 'react-admin';
import { Route } from 'react-router-dom';
import { dataProvider } from './dataProvider';
import { authProvider } from './authProvider';

// Import components
import SystemMetrics from './components/dashboard/SystemMetrics';
import ConfigurationForm from './components/configuration/ConfigurationForm';
import JobList from './components/jobs/JobList';

const Dashboard = () => <SystemMetrics />;

const App = () => (
    <Admin
        dataProvider={dataProvider}
        authProvider={authProvider}
        dashboard={Dashboard}
        title="MCP Dashboard"
    >
        <Resource 
            name="configuration" 
            list={ConfigurationForm}
            edit={ConfigurationForm}
        />
        <Resource 
            name="jobs" 
            list={JobList}
        />
        
        <CustomRoutes>
            <Route path="/metrics" element={<SystemMetrics />} />
        </CustomRoutes>
    </Admin>
);

export default App;
```

## Key Integration Points

1. **Real-time Updates**: WebSocket integration for live job progress and metrics
2. **Configuration Management**: Form-based interface for all environment variables
3. **Job Monitoring**: Visual job queue with progress tracking
4. **System Monitoring**: Real-time metrics and service health
5. **Material-UI Integration**: Consistent design system
6. **Responsive Design**: Mobile-friendly interface
7. **Error Handling**: Proper error states and user feedback

## Performance Optimizations

1. **Lazy Loading**: Code splitting for large components
2. **Memoization**: React.memo for expensive components
3. **WebSocket Management**: Proper connection cleanup
4. **Data Caching**: React Query integration for API caching
5. **Virtualization**: For large data lists

This pattern provides a robust foundation for building a production-ready admin dashboard that integrates seamlessly with the FastAPI backend.