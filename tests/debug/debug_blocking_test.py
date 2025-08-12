#!/usr/bin/env python3
"""
Teste de debug específico para identificar onde está o bloqueio do event loop
durante a execução da ferramenta index_github_repository.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_context():
    """Criar contexto de teste com ThreadPoolExecutor."""
    context = Mock()
    context.request_context = Mock()
    context.request_context.lifespan_context = Mock()

    # Mock components with actual ThreadPoolExecutor
    context.request_context.lifespan_context.qdrant_client = Mock()
    context.request_context.lifespan_context.cpu_executor = ThreadPoolExecutor(
        max_workers=4, thread_name_prefix="debug_test"
    )

    logger.info(
        f"✅ Context created with executor: {context.request_context.lifespan_context.cpu_executor}"
    )
    return context


async def simulate_cpu_work():
    """Simular trabalho CPU-intensivo para detectar bloqueio."""
    logger.info("🔄 Starting CPU-intensive work simulation")

    # Simular trabalho CPU-bound
    start_time = time.time()
    result = 0
    while time.time() - start_time < 1.0:  # 1 segundo de trabalho CPU
        result += sum(range(1000))

    logger.info(f"✅ CPU work completed (result: {result})")
    return result


async def test_event_loop_responsiveness():
    """Testar se o event loop permanece responsivo."""
    response_times = []

    for i in range(10):
        start_time = asyncio.get_event_loop().time()
        await asyncio.sleep(0.1)  # 100ms
        end_time = asyncio.get_event_loop().time()

        response_time = end_time - start_time
        response_times.append(response_time)

        if response_time > 0.2:  # Se demorou mais que 200ms, há bloqueio
            logger.warning(f"⚠️ Event loop blocking detected: {response_time:.3f}s")
        else:
            logger.info(f"✅ Event loop responsive: {response_time:.3f}s")

    avg_response = sum(response_times) / len(response_times)
    logger.info(f"📊 Average response time: {avg_response:.3f}s")

    return avg_response < 0.15  # Considerar OK se menor que 150ms


async def test_with_actual_index_tool():
    """Testar com a ferramenta real index_github_repository."""
    logger.info("🧪 Testing with actual index_github_repository tool")

    try:
        from src.tools.github_tools import index_github_repository
        from src.services.unified_indexing_service import UnifiedIndexingService

        # Criar contexto de teste
        ctx = create_test_context()

        # Mock GitHub processor para evitar operações reais de git
        with patch(
            "src.services.unified_indexing_service.GitHubProcessor"
        ) as mock_github_proc_class:
            mock_github_proc = Mock()

            # Mock clone operation com trabalho CPU-intensivo
            def blocking_clone(*args, **kwargs):
                logger.info(
                    "🔄 Starting repository cloning (potentially blocking operation)"
                )

                # Simular trabalho CPU-bound que DEVERIA estar no ThreadPoolExecutor
                start_time = time.time()
                result = 0
                while time.time() - start_time < 0.5:  # 500ms de trabalho CPU
                    result += sum(range(1000))

                logger.info(f"✅ Repository cloning completed (result: {result})")
                return {"success": True, "temp_directory": "/tmp/test_repo"}

            mock_github_proc.clone_repository_temp = Mock(side_effect=blocking_clone)
            mock_github_proc_class.return_value = mock_github_proc

            # Mock file discovery para retornar lista vazia
            with patch.object(
                UnifiedIndexingService, "_discover_repository_files", return_value=[]
            ):
                # Executar indexação e teste de responsividade concorrentemente
                logger.info(
                    "🚀 Starting concurrent test: indexing + responsiveness check"
                )

                indexing_task = asyncio.create_task(
                    index_github_repository(
                        ctx,
                        repo_url="https://github.com/test/debug-repo",
                        destination="qdrant",
                        file_types=[".py"],
                        max_files=5,
                    )
                )

                responsiveness_task = asyncio.create_task(
                    test_event_loop_responsiveness()
                )

                # Executar ambas as tarefas
                start_time = time.time()
                indexing_result, is_responsive = await asyncio.gather(
                    indexing_task, responsiveness_task
                )
                end_time = time.time()

                total_time = end_time - start_time

                # Analisar resultados
                try:
                    indexing_data = json.loads(indexing_result)
                    indexing_success = indexing_data.get("success", False)
                except:
                    indexing_success = False

                logger.info("=" * 60)
                logger.info("📋 DEBUG TEST RESULTS:")
                logger.info(f"   Total execution time: {total_time:.3f}s")
                logger.info(
                    f"   Repository indexing: {'✅ SUCCESS' if indexing_success else '❌ FAILED'}"
                )
                logger.info(
                    f"   Event loop responsive: {'✅ YES' if is_responsive else '❌ BLOCKED'}"
                )

                if not is_responsive:
                    logger.error("🚨 EVENT LOOP IS BEING BLOCKED!")
                    logger.error(
                        "   The ThreadPoolExecutor integration is NOT working correctly"
                    )
                    logger.error(
                        "   CPU-bound operations are running on the main thread"
                    )
                else:
                    logger.info("🎉 Event loop remained responsive!")
                    logger.info(
                        "   ThreadPoolExecutor integration is working correctly"
                    )

                return is_responsive

    except Exception as e:
        logger.error(f"❌ Error in debug test: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


async def test_isolated_unified_service():
    """Testar isoladamente o UnifiedIndexingService."""
    logger.info("🔬 Testing UnifiedIndexingService in isolation")

    try:
        from src.services.unified_indexing_service import UnifiedIndexingService
        from src.models.unified_indexing_models import (
            UnifiedIndexingRequest,
            IndexingDestination,
        )

        # Criar contexto com executor
        ctx = create_test_context()
        executor = ctx.request_context.lifespan_context.cpu_executor

        # Criar service com executor do contexto
        service = UnifiedIndexingService(qdrant_client=Mock(), cpu_executor=executor)

        logger.info(f"🔧 Service created with executor: {service.cpu_executor}")
        logger.info(f"🔧 Service has own executor: {service._own_executor}")

        # Verificar se está usando o executor correto
        if service.cpu_executor is executor:
            logger.info("✅ Service is using context executor correctly")
        else:
            logger.error("❌ Service is NOT using context executor!")

        # Mock GitHub processor
        with patch(
            "src.services.unified_indexing_service.GitHubProcessor"
        ) as mock_github_proc_class:
            mock_github_proc = Mock()

            def slow_clone(*args, **kwargs):
                logger.info("🔄 UnifiedIndexingService: Starting clone operation")
                # Simular trabalho que deve estar no ThreadPoolExecutor
                import time

                start = time.time()
                result = 0
                while time.time() - start < 0.3:  # 300ms
                    result += sum(range(500))
                logger.info("✅ UnifiedIndexingService: Clone operation completed")
                return {"success": True, "temp_directory": "/tmp/test"}

            mock_github_proc.clone_repository_temp = Mock(side_effect=slow_clone)
            mock_github_proc_class.return_value = mock_github_proc

            # Mock file discovery
            with patch.object(service, "_discover_repository_files", return_value=[]):
                # Criar request
                request = UnifiedIndexingRequest(
                    repo_url="https://github.com/test/isolated-test",
                    destination=IndexingDestination.QDRANT,
                    file_types=[".py"],
                    max_files=5,
                )

                # Executar processamento e teste de responsividade
                processing_task = asyncio.create_task(
                    service.process_repository_unified(request)
                )
                responsiveness_task = asyncio.create_task(
                    test_event_loop_responsiveness()
                )

                response, is_responsive = await asyncio.gather(
                    processing_task, responsiveness_task
                )

                logger.info("📋 ISOLATED SERVICE TEST RESULTS:")
                logger.info(
                    f"   Processing success: {'✅ YES' if response.success else '❌ NO'}"
                )
                logger.info(
                    f"   Event loop responsive: {'✅ YES' if is_responsive else '❌ BLOCKED'}"
                )

                await service.cleanup()
                return is_responsive

    except Exception as e:
        logger.error(f"❌ Error in isolated service test: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


async def main():
    """Função principal de teste."""
    logger.info("🧪 STARTING COMPREHENSIVE DEBUG TEST FOR EVENT LOOP BLOCKING")
    logger.info("=" * 80)

    try:
        # Teste 1: Ferramenta index_github_repository completa
        logger.info("🔍 TEST 1: Complete index_github_repository tool")
        test1_result = await test_with_actual_index_tool()

        logger.info("")

        # Teste 2: UnifiedIndexingService isolado
        logger.info("🔍 TEST 2: Isolated UnifiedIndexingService")
        test2_result = await test_isolated_unified_service()

        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 FINAL DEBUG RESULTS:")
        logger.info(
            f"   Complete tool test: {'✅ PASSED' if test1_result else '❌ FAILED'}"
        )
        logger.info(
            f"   Isolated service test: {'✅ PASSED' if test2_result else '❌ FAILED'}"
        )

        if test1_result and test2_result:
            logger.info("🎉 ALL TESTS PASSED - Event loop blocking is FIXED!")
            return True
        else:
            logger.error("🚨 EVENT LOOP BLOCKING DETECTED!")
            if not test1_result:
                logger.error("   - Complete tool is blocking event loop")
            if not test2_result:
                logger.error("   - UnifiedIndexingService is blocking event loop")
            return False

    except Exception as e:
        logger.error(f"❌ Debug test failed: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())

    print()
    if success:
        print("✅ Event loop blocking has been FIXED!")
        exit(0)
    else:
        print("❌ Event loop is still being BLOCKED!")
        exit(1)
