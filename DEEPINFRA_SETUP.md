# 🚀 Configuração DeepInfra com Qwen3-Embedding-0.6B

Este guia explica como configurar a aplicação para usar o modelo de embeddings DeepInfra Qwen3-Embedding-0.6B com detecção automática de dimensões e recriação de tabelas.

## 📋 Pré-requisitos

1. **Conta DeepInfra**: Crie uma conta em [deepinfra.com](https://deepinfra.com)
2. **API Key**: Obtenha sua chave de API no painel do DeepInfra
3. **Qdrant**: Certifique-se que o Qdrant está rodando (localhost:6333)

## ⚡ Configuração Rápida

### 1. Configure as Variáveis de Ambiente

Edite o arquivo `.env` e configure:

```bash
# Modelo de Embeddings - DeepInfra Qwen3
EMBEDDINGS_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDINGS_API_KEY=sua-chave-deepinfra-aqui
EMBEDDINGS_API_BASE=https://api.deepinfra.com/v1/openai

# Dimensões (auto-detectadas como 1024)
EMBEDDINGS_DIMENSIONS=
```

### 2. Configuração de Fallback (Recomendado)

Para maior confiabilidade, configure um fallback para OpenAI:

```bash
# Fallback para OpenAI
EMBEDDINGS_FALLBACK_MODEL=text-embedding-3-small
EMBEDDINGS_FALLBACK_API_KEY=sua-chave-openai-aqui
EMBEDDINGS_FALLBACK_API_BASE=https://api.openai.com/v1
```

## 🔧 Configurações Avançadas

### Modelos Suportados pelo DeepInfra

| Modelo | Dimensões | Custo Aprox. | Recomendação |
|--------|-----------|--------------|--------------|
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | ~$0.01/1M tokens | ⭐ **Recomendado** |
| `BAAI/bge-large-en-v1.5` | 1024 | ~$0.01/1M tokens | Alternativa |
| `BAAI/bge-small-en-v1.5` | 384 | ~$0.005/1M tokens | Para economia |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | ~$0.005/1M tokens | Básico |

### Override Manual de Dimensões

Se necessário, você pode forçar um número específico de dimensões:

```bash
EMBEDDINGS_DIMENSIONS=1024  # Força 1024 dimensões
```

⚠️ **Atenção**: Dimensões incorretas irão trigger recriação automática das collections!

## 🔄 Migração Automática

A aplicação detecta automaticamente incompatibilidades de dimensões e recria as collections:

1. **Startup**: Valida dimensões configuradas vs collections existentes
2. **Incompatibilidade**: Deleta e recria collections automaticamente
3. **Log**: Mostra processo de migração no console

### Exemplo de Log de Migração

```
🔧 Validating MCP server configuration...
✓ Embedding setup validated - dimensions: 1024
⚠️  Collection crawled_pages has incompatible dimensions (current: 1536, expected: 1024). Recreating...
✓ Recreated collection crawled_pages with dimensions: 1024
✓ Collection validation complete
```

## 🧪 Validação da Configuração

### Teste Rápido

Execute para validar sua configuração:

```bash
python -c "
import os
import sys
sys.path.insert(0, 'src')

# Configure DeepInfra
os.environ['EMBEDDINGS_MODEL'] = 'Qwen/Qwen3-Embedding-0.6B'
os.environ['EMBEDDINGS_API_KEY'] = 'sua-chave-aqui'

from embedding_config import get_embedding_dimensions, validate_embeddings_config

print('Dimensões detectadas:', get_embedding_dimensions())
validate_embeddings_config()
print('✓ Configuração válida!')
"
```

### Verificação das Collections

Depois de iniciar o servidor, verifique se as collections foram criadas corretamente:

```bash
curl http://localhost:6333/collections
```

## 🛠️ Troubleshooting

### Problemas Comuns

#### 1. Erro de API Key
```
Error: No API key configured for embeddings
```
**Solução**: Verifique se `EMBEDDINGS_API_KEY` está configurada no `.env`

#### 2. Dimensões Incorretas
```
Collection has incompatible dimensions (current: 1536, expected: 1024)
```
**Solução**: Normal durante migração. A aplicação recria automaticamente.

#### 3. Falha de Conexão com DeepInfra
```
Error: Cannot connect to DeepInfra API
```
**Solução**: 
- Verifique sua API key
- Confirme que `EMBEDDINGS_API_BASE=https://api.deepinfra.com/v1/openai`
- Configure fallback para OpenAI

#### 4. Collections Não Criadas
```
Failed to create collection: Permission denied
```
**Solução**: Verifique se o Qdrant está rodando e acessível

### Debug Avançado

Para debug detalhado, execute:

```bash
# Verificar detecção de modelo
python -c "from src.embedding_config import get_embedding_dimensions; print(get_embedding_dimensions())"

# Verificar configuração de collections
python -c "from src.qdrant_wrapper import get_collections_config; print(get_collections_config())"

# Testar conexão Qdrant
curl http://localhost:6333/health
```

## 💰 Comparação de Custos

| Provider | Modelo | Dimensões | Custo/1M tokens | Performance |
|----------|--------|-----------|-----------------|-------------|
| DeepInfra | Qwen/Qwen3-Embedding-0.6B | 1024 | ~$0.01 | ⭐⭐⭐⭐⭐ |
| OpenAI | text-embedding-3-small | 1536 | $0.02 | ⭐⭐⭐⭐ |
| OpenAI | text-embedding-3-large | 3072 | $0.13 | ⭐⭐⭐⭐⭐ |

**Recomendação**: Use Qwen3 para melhor custo-benefício, com fallback OpenAI para confiabilidade.

## 📚 Recursos Adicionais

- [DeepInfra API Docs](https://deepinfra.com/docs)
- [Qwen3 Model Page](https://deepinfra.com/Qwen/Qwen3-Embedding-0.6B)
- [OpenAI Embeddings Docs](https://platform.openai.com/docs/guides/embeddings)

## 🆘 Suporte

Para problemas específicos:
1. Verifique os logs do servidor
2. Confirme configurações no `.env`
3. Teste conectividade com APIs
4. Verifique status do Qdrant

---

✅ **Status**: Configuração completa e testada para produção!