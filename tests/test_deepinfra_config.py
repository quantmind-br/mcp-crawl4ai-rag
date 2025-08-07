#!/usr/bin/env python3
# ruff: noqa: E402
"""
Teste rápido da configuração DeepInfra Qwen3-Embedding-0.6B

Este script valida que a configuração DeepInfra está funcionando corretamente.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def test_deepinfra_config():
    """Testa a configuração completa do DeepInfra."""

    print("=== Testando Configuracao DeepInfra Qwen3-Embedding-0.6B ===")
    print("=" * 60)

    try:
        # 1. Teste de detecção de dimensões
        print("\n1. Testando detecção de dimensões...")
        from embedding_config import get_embedding_dimensions
        from embedding_config import reset_embeddings_config

        reset_embeddings_config()

        # Simular configuração DeepInfra
        os.environ["EMBEDDINGS_MODEL"] = "Qwen/Qwen3-Embedding-0.6B"
        os.environ.pop("EMBEDDINGS_DIMENSIONS", None)  # Remove override

        dims = get_embedding_dimensions()
        print(f"   OK Dimensoes detectadas: {dims}")
        assert dims == 1024, f"Esperado 1024, obtido {dims}"

        # 2. Teste de configuração de collections
        print("
2. Testando configuração de collections...")
        # We need to reset because get_embedding_dimensions caches the result
        reset_embeddings_config()
        from src.clients.qdrant_client import get_collections_config

        config = get_collections_config()
        crawled_dims = config["crawled_pages"]["vectors_config"].size
        code_dims = config["code_examples"]["vectors_config"].size

        print(f"   ✓ Collection 'crawled_pages': {crawled_dims} dimensões")
        print(f"   ✓ Collection 'code_examples': {code_dims} dimensões")

        assert crawled_dims == 1024, (
            f"crawled_pages: esperado 1024, obtido {crawled_dims}"
        )
        assert code_dims == 1024, f"code_examples: esperado 1024, obtido {code_dims}"

        # 3. Teste de override manual
        print("
3. Testando override manual de dimensões...")
        reset_embeddings_config()
        os.environ["EMBEDDINGS_DIMENSIONS"] = "512"

        dims_override = get_embedding_dimensions()
        print(f"   ✓ Override funcionando: {dims_override} dimensões")
        assert dims_override == 512, (
            f"Override falhou: esperado 512, obtido {dims_override}"
        )

        # 4. Teste de validação
        print("
4. Testando validação de configuração...")
        reset_embeddings_config()
        from embedding_config import validate_embeddings_config

        # Simular API key
        os.environ["EMBEDDINGS_API_KEY"] = "test-key-for-validation"

        try:
            validate_embeddings_config()
            print("   ✓ Validação de configuração funcionando")
        except ValueError as e:
            if "API key" in str(e):
                print("   ✓ Validação detectou API key corretamente")
            else:
                raise

        # 5. Teste de modelos OpenAI (backward compatibility)
        print("
5. Testando compatibilidade com modelos OpenAI...")
        reset_embeddings_config()
        os.environ["EMBEDDINGS_MODEL"] = "text-embedding-3-small"
        os.environ.pop("EMBEDDINGS_DIMENSIONS", None)

        openai_dims = get_embedding_dimensions()
        print(f"   ✓ OpenAI text-embedding-3-small: {openai_dims} dimensões")
        assert openai_dims == 1536, f"OpenAI: esperado 1536, obtido {openai_dims}"

        # 6. Teste de modelo desconhecido (deve falhar)
        print("
6. Testando erro para modelo desconhecido...")
        reset_embeddings_config()
        os.environ["EMBEDDINGS_MODEL"] = "modelo-inexistente"
        os.environ.pop("EMBEDDINGS_DIMENSIONS", None)

        with pytest.raises(ValueError, match="Unknown embeddings model"):
            get_embedding_dimensions()
        print("   ✓ Erro esperado para modelo desconhecido, OK")

        print("\n" + "=" * 60)
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("\n✅ Configuração DeepInfra está funcionando perfeitamente!")
        print("\n📝 Próximos passos:")
        print("   1. Configure sua EMBEDDINGS_API_KEY no .env")
        print("   2. Inicie o servidor MCP")
        print("   3. As collections serão criadas/migradas automaticamente")

        return True

    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Verifique se o diretório 'src' existe")
        print("   2. Confirme que os arquivos de configuração estão presentes")
        print("   3. Verifique a estrutura do projeto")
        return False


def check_env_file():
    """Verifica se o arquivo .env está configurado corretamente."""

    print("\n📁 Verificando arquivo .env...")

    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        print("   ⚠️  Arquivo .env não encontrado")
        print("   💡 Copie .env.example para .env e configure suas chaves")
        return False

    # Lê o .env
    with open(env_file, "r", encoding="utf-8") as f:
        content = f.read()

    checks = {
        "EMBEDDINGS_MODEL": "Qwen/Qwen3-Embedding-0.6B" in content,
        "EMBEDDINGS_API_BASE": "deepinfra.com" in content,
        "EMBEDDINGS_API_KEY": "EMBEDDINGS_API_KEY=" in content,
    }

    for key, found in checks.items():
        status = "✓" if found else "⚠️"
        print(f"   {status} {key}: {'Configurado' if found else 'Precisa configurar'}")

    if all(checks.values()):
        print("   🎯 Arquivo .env está configurado para DeepInfra!")
        return True
    else:
        print("   📝 Configure as variáveis necessárias no .env")
        return False


if __name__ == "__main__":
    print("=== Verificador de Configuracao DeepInfra ===")
    print("Validando integracao com Qwen3-Embedding-0.6B")

    # Verificar .env
    env_ok = check_env_file()

    # Executar testes
    tests_ok = test_deepinfra_config()

    if tests_ok and env_ok:
        print("\n🚀 Sistema pronto para usar DeepInfra!")
        sys.exit(0)
    else:
        print("\n⚠️  Configure o sistema antes de continuar")
        sys.exit(1)
