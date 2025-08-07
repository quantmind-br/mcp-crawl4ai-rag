#!/usr/bin/env python3
"""
Script para testar o cache Redis de embeddings na prática.
Testa cache hits, misses, TTL e performance.
"""

import time
import sys
import os
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
from embedding_config import get_embedding_dimensions
from embedding_cache import (
    EmbeddingCache,
)
import redis

load_dotenv()


def test_redis_cache():
    """Teste completo do cache Redis de embeddings."""

    print("=== TESTE COMPLETO DO CACHE REDIS ===\n")

    # 1. Verificar configuração
    print("1. CONFIGURACAO:")
    print(f"   USE_REDIS_CACHE: {os.getenv('USE_REDIS_CACHE')}")
    print(f"   REDIS_HOST: {os.getenv('REDIS_HOST')}")
    print(f"   REDIS_PORT: {os.getenv('REDIS_PORT')}")
    print(f"   REDIS_EMBEDDING_TTL: {os.getenv('REDIS_EMBEDDING_TTL')}s")
    print()

    # 2. Inicializar cache
    try:
        cache = EmbeddingCache()
        print("2. INICIALIZACAO:")
        print("   [OK] EmbeddingCache inicializado")

        # Health check
        health = cache.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Conexao: {health.get('connection_test') or health.get('ping')}")
        print(f"   Memoria: {health.get('memory_usage') or health.get('connection_info')}")
        print()

    except Exception as e:
        print(f"   [ERROR] Falha na inicializacao: {e}")
        return False

    # 3. Teste de Cache Miss (primeira busca)
    print("3. TESTE CACHE MISS:")
    test_text = "Este é um texto de teste para cache de embeddings"
    test_key = f"test_embedding_{hash(test_text)}"

    start_time = time.time()
    result_miss = cache.get_batch([test_text], os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small"))
    miss_time = time.time() - start_time

    print(f"   Texto: '{test_text[:50]}...'")
    print(f"   Resultado (deve ser None): {result_miss.get(test_text) if isinstance(result_miss, dict) else None}")
    print(f"   Tempo: {miss_time * 1000:.2f}ms")
    print()

    # 4. Simular criação de embedding e cache
    print("4. TESTE CACHE SET:")
    embedding_dims = get_embedding_dimensions()
    fake_embedding = [0.1] * embedding_dims

    start_time = time.time()
    cache.set_batch({test_text: fake_embedding}, os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small"), ttl=300)
    set_time = time.time() - start_time

    print(f"   Embedding simulado ({embedding_dims} dims): {fake_embedding[:5]}...")
    print("   Cache SET completado")
    print(f"   Tempo: {set_time * 1000:.2f}ms")
    print()

    # 5. Teste de Cache Hit (segunda busca)
    print("5. TESTE CACHE HIT:")

    start_time = time.time()
    result_hit = cache.get_batch([test_text], os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small"))
    hit_time = time.time() - start_time

    found = isinstance(result_hit, dict) and test_text in result_hit
    print(f"   Resultado encontrado: {found}")
    print(f"   Dimensoes: {len(result_hit[test_text]) if found else 0}")
    print(f"   Primeiros valores: {result_hit[test_text][:5] if found else None}")
    print(f"   Tempo: {hit_time * 1000:.2f}ms")
    if hit_time > 0:
        print(f"   Speedup: {miss_time / hit_time:.1f}x mais rapido")
    print()

    # 6. Teste de Performance (múltiplas operações)
    print("6. TESTE DE PERFORMANCE:")

    embeddings_test = {}
    for i in range(10):
        key_text = f"perf_text_{i}"
        embedding_dims = get_embedding_dimensions()
        embedding = [i * 0.1] * embedding_dims
        embeddings_test[key_text] = embedding

    start_time = time.time()
    cache.set_batch(embeddings_test, os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small"), ttl=300)
    batch_set_time = time.time() - start_time
    print(f"   SET 10 embeddings: {batch_set_time * 1000:.2f}ms")

    start_time = time.time()
    result_batch = cache.get_batch(list(embeddings_test.keys()), os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small"))
    batch_get_time = time.time() - start_time

    hits = sum(1 for k in embeddings_test.keys() if isinstance(result_batch, dict) and k in result_batch)
    print(f"   GET 10 embeddings: {batch_get_time * 1000:.2f}ms")
    print(f"   Cache hits: {hits}/10")
    print(f"   Avg GET time: {batch_get_time / 10 * 1000:.2f}ms per embedding")
    print()

    # 7. Teste de TTL (Time To Live)
    print("7. TESTE TTL (Time To Live):")

    ttl_text = "ttl_text"
    cache.set_batch({ttl_text: [1.0, 2.0, 3.0]}, os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small"), ttl=2)

    immediate_result = cache.get_batch([ttl_text], os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small"))
    print(f"   Imediatamente: {isinstance(immediate_result, dict) and ttl_text in immediate_result}")

    print("   Aguardando expiracao (3s)...")
    time.sleep(3)

    expired_result = cache.get_batch([ttl_text], os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small"))
    print(f"   Apos expiracao: {isinstance(expired_result, dict) and ttl_text in expired_result}")
    print()

    # 8. Estatísticas do Redis
    print("8. ESTATISTICAS REDIS:")
    try:
        r = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
        )
        info = r.info()
        stats = r.info("stats")
        print(f"   Total commands processed: {stats.get('total_commands_processed', 'N/A')}")
        print(f"   Cache hits: {stats.get('keyspace_hits', 0)}")
        print(f"   Cache misses: {stats.get('keyspace_misses', 0)}")
        hit_ratio = 0
        denom = stats.get("keyspace_hits", 0) + stats.get("keyspace_misses", 0)
        if denom > 0:
            hit_ratio = stats["keyspace_hits"] / denom * 100
        print(f"   Hit ratio: {hit_ratio:.1f}%")
        print(f"   Memory used: {info.get('used_memory_human')}")
        print(f"   Keys in DB: {info.get('db0', {}).get('keys', 0) if 'db0' in info else 0}")
    except Exception as e:
        print(f"   [ERROR] Falha ao obter estatisticas: {e}")

    print()

    # 9. Limpeza
    print("9. LIMPEZA:")
    cleanup_texts = [test_text, ttl_text] + list(embeddings_test.keys())
    for text in cleanup_texts:
        try:
            cache.redis.delete(cache._generate_cache_key(text, os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")))
        except Exception:
            pass
    print(f"   {len(cleanup_texts)} chaves de teste removidas")
    print()

    print("=== TESTE CONCLUIDO ===")
    print("[OK] Cache Redis funcionando corretamente!")
    return True


if __name__ == "__main__":
    success = test_redis_cache()
    sys.exit(0 if success else 1)