# Test Organization Guide

Esta documentação descreve a estrutura organizada dos testes do projeto MCP Crawl4AI RAG.

## 📁 Estrutura de Diretórios

### 🏗️ `/unit/` - Testes Unitários
Testes isolados de componentes individuais, organizados por módulo:

- **`/unit/clients/`** - Clientes de API e conexões externas
  - `test_llm_api_client.py` - Cliente OpenAI/LLM
  - `test_qdrant_client.py` - Cliente Qdrant
  - `test_fallback_api_config.py` - Configuração de fallback de APIs
  - `test_flexible_api_config.py` - Sistema de configuração flexível

- **`/unit/core/`** - Funcionalidades centrais da aplicação
  - `test_core_app.py` - Aplicação MCP principal
  - `test_core_context.py` - Gerenciamento de contexto

- **`/unit/services/`** - Serviços de negócio
  - `test_rag_service.py` - Serviço RAG (Retrieval Augmented Generation)
  - `test_unified_indexing_service.py` - Serviço de indexação unificada
  - `test_neo4j_batch_processing.py` - Processamento em lote Neo4j

- **`/unit/tools/`** - Ferramentas MCP
  - `test_web_tools.py` - Ferramentas de crawling web
  - `test_github_tools.py` - Ferramentas GitHub
  - `test_rag_tools.py` - Ferramentas RAG
  - `test_kg_tools.py` - Ferramentas knowledge graph

- **`/unit/utils/`** - Utilitários e helpers
  - `test_validation.py` - Funções de validação
  - `test_file_id_generator.py` - Geração de IDs de arquivo
  - `test_grammar_initialization.py` - Inicialização de gramáticas
  - `test_backward_compatibility.py` - Compatibilidade retroativa
  - `test_file_id_linking.py` - Linkagem de IDs entre sistemas

### 🧪 `/specialized/` - Testes Especializados
Testes de funcionalidades específicas e complexas:

- **`/specialized/embedding/`** - Sistema de embeddings
  - `test_embedding_service.py` - Serviço principal de embeddings
  - `test_embedding_cache.py` - Cache de embeddings
  - `test_embedding_config.py` - Configuração de embeddings
  - `test_sparse_vector_types.py` - Tipos de vetores esparsos

- **`/specialized/knowledge_graphs/`** - Grafos de conhecimento
  - `test_knowledge_graph_validator.py` - Validação de KG
  - `test_language_parser.py` - Parser de linguagens
  - `test_parser_factory.py` - Factory de parsers
  - `test_query_knowledge_graph.py` - Consultas ao KG
  - `test_tree_sitter_parser.py` - Parser Tree-sitter

- **`/specialized/device_management/`** - Gerenciamento de dispositivos
  - `test_device_manager.py` - Gerenciador de dispositivos GPU/CPU
  - `test_event_loop_fix.py` - Correções de event loop
  - `test_event_loop_fix_comprehensive.py` - Testes abrangentes de event loop

### 🏗️ `/infrastructure/` - Testes de Infraestrutura
Testes de componentes de infraestrutura:

- **`/infrastructure/storage/`** - Sistemas de armazenamento
  - `test_qdrant_optimization.py` - Otimizações Qdrant
  - `test_qdrant_wrapper.py` - Wrapper Qdrant
  - `test_redis_cache.py` - Cache Redis
  - `test_redis_integration.py` - Integração Redis

- **`/infrastructure/validation/`** - Validação de dados
  - `test_validation_functions.py` - Funções de validação

### 🔗 `/integration/` - Testes de Integração
Testes que verificam a interação entre múltiplos componentes:

- `test_integration_basic.py` - Integração básica end-to-end
- `test_concurrent_clients.py` - Clientes concorrentes
- `test_hybrid_rag.py` - RAG híbrido (vetorial + sparse)

### ⚡ `/performance/` - Testes de Performance
Testes de benchmarks e performance (atualmente vazio após limpeza)

### 📋 `/fixtures/` - Arquivos de Teste
Dados de exemplo para testes multi-linguagem:
- `sample.py`, `sample.js`, `sample.go`, etc. - Exemplos de código para parsers

## 🎯 Convenções de Nomenclatura

### ✅ Nomes Padronizados
- **Arquivos de teste**: `test_*.py`
- **Classes de teste**: `Test*` ou `*Test`
- **Métodos de teste**: `test_*`

### 📝 Estrutura de Classes
```python
class TestComponentName:
    """Test suite for ComponentName functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        pass
    
    def test_specific_functionality(self):
        """Test specific functionality with descriptive name."""
        pass
```

## 🏃‍♂️ Executando Testes

### Todos os testes
```bash
uv run pytest
```

### Por categoria
```bash
# Testes unitários
uv run pytest tests/unit/

# Testes especializados
uv run pytest tests/specialized/

# Testes de integração
uv run pytest tests/integration/

# Testes de infraestrutura
uv run pytest tests/infrastructure/
```

### Por módulo específico
```bash
# Testes de embedding
uv run pytest tests/specialized/embedding/

# Testes de clientes
uv run pytest tests/unit/clients/

# Testes de ferramentas
uv run pytest tests/unit/tools/
```

### Com cobertura
```bash
uv run pytest --cov=src --cov-report=html
```

## 📊 Benefícios da Nova Organização

1. **🎯 Navegação Intuitiva**: Estrutura hierárquica clara por funcionalidade
2. **🚀 Execução Seletiva**: Facilita execução de grupos específicos de testes
3. **📈 Manutenibilidade**: Localização rápida de testes relevantes
4. **🔄 Escalabilidade**: Estrutura preparada para crescimento do projeto
5. **👥 Colaboração**: Padronização que facilita contribuições da equipe

## 🔧 Configuração

- **`conftest.py`**: Fixtures compartilhadas e configuração pytest
- **`__init__.py`**: Arquivos de módulo Python em cada diretório
- **Dependências**: Definidas em `pyproject.toml`

---

*Estrutura organizada em 2025 para otimizar desenvolvimento e manutenção dos testes.*