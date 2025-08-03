# Prompt: Correção de ConnectionResetError no Windows Asyncio

## **Contexto do Projeto**

Você está trabalhando em um servidor MCP (Model Context Protocol) para Crawl4AI com integração RAG e Qdrant. O projeto usa Python 3.12+ no Windows com as seguintes tecnologias:

- **Framework**: MCP server com FastAPI/Uvicorn
- **HTTP Client**: httpx para requisições à API DeepInfra 
- **Async**: asyncio para operações concorrentes
- **Embeddings**: DeepInfra API (Qwen/Qwen3-Embedding-0.6B)
- **Vector DB**: Qdrant para armazenamento de embeddings

## **Problema Identificado**

**Erro**: `ConnectionResetError: [WinError 10054] Foi forçado o cancelamento de uma conexão existente pelo host remoto`

**Log do Erro**:
```
2025-08-03 06:13:21 - httpx - INFO - HTTP Request: POST https://api.deepinfra.com/v1/openai/embeddings "HTTP/1.1 200 OK"
Inserted batch 19 of 19 code examples
2025-08-03 06:13:22 - asyncio - ERROR - Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)
handle: <Handle _ProactorBasePipeTransport._call_connection_lost(None)>
Traceback (most recent call last):
  File "C:\Users\Diogo\AppData\Local\Programs\Python\Python312\Lib\asyncio\events.py", line 88, in _run
    self._context.run(self._callback, *self._args)
  File "C:\Users\Diogo\AppData\Local\Programs\Python\Python312\Lib\asyncio\proactor_events.py", line 165, in _call_connection_lost
    self._sock.shutdown(socket.SHUT_RDWR)
ConnectionResetError: [WinError 10054] Foi forçado o cancelamento de uma conexão existente pelo host remoto
```

## **Análise Técnica**

- **Timing**: Erro ocorre APÓS operações HTTP bem-sucedidas
- **Causa**: Race condition entre fechamento de socket pelo cliente/servidor
- **Localização**: Windows ProactorEventLoop durante cleanup de conexão
- **Impacto**: Cosmético (funcionalidade não afetada, dados processados corretamente)

## **Estrutura do Projeto**

```
mcp-crawl4ai-rag/
├── src/
│   ├── crawl4ai_mcp.py        # Servidor MCP principal
│   ├── utils.py               # Funções HTTP e embeddings
│   ├── qdrant_wrapper.py      # Cliente Qdrant
│   └── __main__.py            # Entry point
├── pyproject.toml             # Dependências do projeto
└── .env                       # Configuração (APIs, modelos)
```

## **Tarefa**

**OBJETIVO**: Implementar uma correção elegante para eliminar o ConnectionResetError no Windows sem afetar a funcionalidade existente.

**REQUISITOS**:

1. **Corrigir o erro** de ConnectionResetError no Windows
2. **Preservar funcionalidade** existente (todas as operações HTTP devem continuar funcionando)
3. **Manter compatibilidade** com Linux/macOS se aplicável
4. **Implementação limpa** que não introduza complexidade desnecessária
5. **Solução robusta** que funcione com diferentes versões do Python 3.12+

**OPÇÕES DE SOLUÇÃO** (escolha a mais apropriada):

- **Event Loop Policy**: Usar SelectorEventLoop no Windows
- **Exception Handling**: Suprimir erros específicos de cleanup
- **HTTP Client Config**: Ajustar timeouts e configurações httpx
- **Connection Management**: Melhorar gerenciamento de conexões

**CRITÉRIOS DE AVALIAÇÃO**:

- ✅ Elimina o erro sem quebrar funcionalidade
- ✅ Implementação enxuta e elegante  
- ✅ Compatibilidade cross-platform
- ✅ Sem impacto na performance
- ✅ Fácil manutenção futura

**DELIVERABLES**:

1. Código da correção implementada
2. Explicação da solução escolhida
3. Instruções de teste para verificar a correção
4. Documentação de qualquer mudança de comportamento

**CONTEXTO ADICIONAL**: O sistema está funcionando corretamente - este é puramente um fix de qualidade de vida para eliminar logs de erro desnecessários que podem confundir usuários e mascarar problemas reais.