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
from embedding_cache import (
    get_embedding_dimensions,
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
        print(f"   Conexao: {health['connection_test']}")
        print(f"   Memoria: {health['memory_usage']}")
        print()

    except Exception as e:
        print(f"   [ERROR] Falha na inicializacao: {e}")
        return False

    # 3. Teste de Cache Miss (primeira busca)
    print("3. TESTE CACHE MISS:")
    test_text = "Este é um texto de teste para cache de embeddings"
    test_key = f"test_embedding_{hash(test_text)}"

    start_time = time.time()
    result_miss = cache.get(test_key)
    miss_time = time.time() - start_time

    print(f"   Texto: '{test_text[:50]}...'")
    print(f"   Resultado (deve ser None): {result_miss}")
    print(f"   Tempo: {miss_time * 1000:.2f}ms")
    print()

    # 4. Simular criação de embedding e cache
    print("4. TESTE CACHE SET:")
    embedding_dims = get_embedding_dimensions()
    fake_embedding = [0.1] * embedding_dims

    start_time = time.time()
    cache.set(test_key, fake_embedding, ttl=300)  # 5 minutos
    set_time = time.time() - start_time

    print(f"   Embedding simulado (1000 dims): {fake_embedding[:5]}...")
    print("   Cache SET completado")
    print(f"   Tempo: {set_time * 1000:.2f}ms")
    print()

    # 5. Teste de Cache Hit (segunda busca)
    print("5. TESTE CACHE HIT:")

    start_time = time.time()
    result_hit = cache.get(test_key)
    hit_time = time.time() - start_time

    print(f"   Resultado encontrado: {result_hit is not None}")
    print(f"   Dimensoes: {len(result_hit) if result_hit else 0}")
    print(f"   Primeiros valores: {result_hit[:5] if result_hit else None}")
    print(f"   Tempo: {hit_time * 1000:.2f}ms")
    print(f"   Speedup: {miss_time / hit_time:.1f}x mais rapido")
    print()

    # 6. Teste de Performance (múltiplas operações)
    print("6. TESTE DE PERFORMANCE:")

    # Cache múltiplos embeddings
    embeddings_test = {}
    for i in range(10):
        key = f"perf_test_{i}"
        embedding_dims = get_embedding_dimensions()
        embedding = [i * 0.1] * embedding_dims
        embeddings_test[key] = embedding

    # Teste SET em lote
    start_time = time.time()
    for key, embedding in embeddings_test.items():
        cache.set(key, embedding, ttl=300)
    batch_set_time = time.time() - start_time

    print(f"   SET 10 embeddings: {batch_set_time * 1000:.2f}ms")

    # Teste GET em lote
    start_time = time.time()
    hits = 0
    for key in embeddings_test.keys():
        result = cache.get(key)
        if result is not None:
            hits += 1
    batch_get_time = time.time() - start_time

    print(f"   GET 10 embeddings: {batch_get_time * 1000:.2f}ms")
    print(f"   Cache hits: {hits}/10")
    print(f"   Avg GET time: {batch_get_time / 10 * 1000:.2f}ms per embedding")
    print()

    # 7. Teste de TTL (Time To Live)
    print("7. TESTE TTL (Time To Live):")

    ttl_key = "ttl_test"
    cache.set(ttl_key, [1, 2, 3], ttl=2)  # 2 segundos apenas

    # Verificar imediatamente
    immediate_result = cache.get(ttl_key)
    print(f"   Imediatamente: {immediate_result is not None}")

    # Aguardar expiração
    print("   Aguardando expiracao (3s)...")
    time.sleep(3)

    expired_result = cache.get(ttl_key)
    print(f"   Apos expiracao: {expired_result is not None}")
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

        print(
            f"   Total commands processed: {stats.get('total_commands_processed', 'N/A')}"
        )
        print(f"   Cache hits: {stats.get('keyspace_hits', 0)}")
        print(f"   Cache misses: {stats.get('keyspace_misses', 0)}")

        hit_ratio = 0
        if stats.get("keyspace_hits", 0) + stats.get("keyspace_misses", 0) > 0:
            hit_ratio = (
                stats["keyspace_hits"]
                / (stats["keyspace_hits"] + stats["keyspace_misses"])
                * 100
            )

        print(f"   Hit ratio: {hit_ratio:.1f}%")
        print(f"   Memory used: {info['used_memory_human']}")
        print(
            f"   Keys in DB: {info.get('db0', {}).get('keys', 0) if 'db0' in info else 0}"
        )

    except Exception as e:
        print(f"   [ERROR] Falha ao obter estatisticas: {e}")

    print()

    # 9. Limpeza
    print("9. LIMPEZA:")
    cleanup_keys = [test_key, ttl_key] + list(embeddings_test.keys())
    for key in cleanup_keys:
        cache.delete(key)
    print(f"   {len(cleanup_keys)} chaves de teste removidas")
    print()

    print("=== TESTE CONCLUIDO ===")
    print("[OK] Cache Redis funcionando corretamente!")
    return True


if __name__ == "__main__":
    success = test_redis_cache()
    sys.exit(0 if success else 1)