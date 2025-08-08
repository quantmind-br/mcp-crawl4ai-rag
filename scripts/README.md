# Database Cleanup Scripts

Scripts utilitários para limpeza completa das bases de dados Qdrant e Neo4j do sistema MCP Crawl4AI RAG.

## 📋 Scripts Disponíveis

### 🐍 `cleanup_databases.py`
Script Python principal com funcionalidades completas de limpeza.

### 🪟 `cleanup.bat`  
Script batch para Windows com interface interativa.

### 🐧 `cleanup.sh`
Script shell para Linux/Mac com interface interativa.

## 🚀 Uso Rápido

### Windows
```batch
# Interface interativa
scripts\cleanup.bat

# Comando direto
python scripts\cleanup_databases.py --confirm
```

### Linux/Mac
```bash
# Interface interativa
./scripts/cleanup.sh

# Comando direto
python3 scripts/cleanup_databases.py --confirm
```

## ⚙️ Opções do Script Python

### Básico
```bash
# Limpar ambas as bases (com confirmação)
python scripts/cleanup_databases.py

# Limpar ambas as bases (sem confirmação)
python scripts/cleanup_databases.py --confirm

# Apenas visualizar o que seria deletado
python scripts/cleanup_databases.py --dry-run
```

### Seletivo
```bash
# Limpar apenas Qdrant
python scripts/cleanup_databases.py --qdrant-only

# Limpar apenas Neo4j
python scripts/cleanup_databases.py --neo4j-only

# Combinações
python scripts/cleanup_databases.py --qdrant-only --dry-run
python scripts/cleanup_databases.py --neo4j-only --confirm
```

## 🗃️ O que é Limpo

### Qdrant (Vector Database)
- ✅ **Todas as coleções** (`crawl_documents`, `code_examples`, `sources`)
- ✅ **Todos os documentos** indexados
- ✅ **Todos os vetores** embeddings
- ✅ **Metadados** associados
- ✅ **Recria coleções padrão** automaticamente

### Neo4j (Knowledge Graph)
- ✅ **Todos os nós** (Repository, File, Class, Method, Function, Attribute)
- ✅ **Todos os relacionamentos** (CONTAINS, DEFINES, HAS_METHOD, etc.)
- ✅ **Constraints** personalizados
- ✅ **Índices** personalizados
- ❌ **Não remove** índices de lookup do sistema

## 📊 Estatísticas

O script mostra estatísticas detalhadas:

```
📊 Current database statistics:
  QDRANT: {'collections': 3, 'total_documents': 1547, 'crawl_documents_documents': 1200, 'code_examples_documents': 347}
  NEO4J: {'nodes': 892, 'relationships': 1756, 'Repository_nodes': 2, 'File_nodes': 45, 'Class_nodes': 123, 'Function_nodes': 567}

🧹 Cleanup completed:
  QDRANT: {'collections_deleted': 3, 'documents_deleted': 1547, 'sources_cleared': 0}  
  NEO4J: {'nodes_deleted': 892, 'relationships_deleted': 1756, 'constraints_deleted': 2, 'indexes_deleted': 5}
```

## 🛡️ Recursos de Segurança

### Confirmação Interativa
```bash
⚠️ This will DELETE ALL DATA in both databases. Continue? (yes/no):
```

### Modo Dry Run
```bash
python scripts/cleanup_databases.py --dry-run

# Output:
🔍 [DRY RUN] Would delete collection: crawl_documents
🔍 [DRY RUN] Would delete 892 nodes and 1756 relationships
```

### Logging Detalhado
- ✅ Timestamp em todas as operações
- ✅ Contadores de itens deletados  
- ✅ Tratamento de erros com contexto
- ✅ Warnings para operações que falharam

## 🔧 Variáveis de Ambiente

O script usa as seguintes variáveis de ambiente (com valores padrão):

```env
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Qdrant (configurado via get_qdrant_client())
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## ⚠️ Avisos Importantes

### 🚨 **DADOS PERMANENTEMENTE PERDIDOS**
- Esta operação **NÃO PODE SER DESFEITA**
- Certifique-se de fazer backup se necessário
- Use `--dry-run` primeiro para verificar o que será deletado

### 🔄 **Recriação Automática**
- Qdrant: Coleções padrão são recriadas automaticamente
- Neo4j: Banco fica completamente vazio (precisa reindexar)

### 🐛 **Resolução de Problemas**

### Erros Comuns

**"QdrantClientWrapper object has no attribute 'get_collections'"**
- ✅ **Resolvido** - O script foi corrigido para usar o cliente interno correto
- O sistema usa um wrapper customizado que encapsula o cliente Qdrant real

**Conexão com Neo4j falha**
- Verifique se Neo4j está rodando: `docker ps` ou `systemctl status neo4j`
- Confirme as credenciais nas variáveis de ambiente
- Teste conexão: `python -c "from neo4j import GraphDatabase; print('OK')"`

**Conexão com Qdrant falha**
- Verifique se Qdrant está rodando: `curl http://localhost:6333`
- Confirme a porta nas configurações (padrão: 6333)
- Execute com `--dry-run` para diagnosticar problemas

**Dependências ausentes**
- Execute: `pip install qdrant-client neo4j`
- Certifique-se que está no ambiente virtual correto

## 📝 Exemplos de Uso

### Desenvolvimento - Reset completo
```bash
# Limpar tudo e começar do zero
python scripts/cleanup_databases.py --confirm

# Reindexar repositórios
python -m src.tools.github_tools index_repository https://github.com/user/repo
```

### Produção - Limpeza seletiva  
```bash
# Apenas limpar vetores (manter grafo)
python scripts/cleanup_databases.py --qdrant-only --confirm

# Apenas limpar grafo (manter vetores)
python scripts/cleanup_databases.py --neo4j-only --confirm
```

### Debug - Análise sem alterações
```bash
# Ver estatísticas atuais
python scripts/cleanup_databases.py --dry-run
```

## 🤝 Contribuição

Para melhorar os scripts:

1. Adicione suporte para outras bases de dados
2. Implemente backup automático antes da limpeza  
3. Adicione métricas de tempo de execução
4. Crie interface web para operação remota

---

⚡ **Scripts prontos para uso em produção e desenvolvimento!**