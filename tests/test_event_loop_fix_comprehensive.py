"""
Testes abrangentes para as funcionalidades de event_loop_fix.py.

Este módulo contém testes para o sistema de configuração de event loop, incluindo:
- Detecção de plataforma Windows
- Verificação de disponibilidade de políticas de event loop
- Detecção de importação do Playwright
- Configuração de event loop
- Validação de configuração
- Tratamento de erros
"""

import sys
from unittest.mock import Mock, patch

# Importa as funções do event_loop_fix
try:
    from src.event_loop_fix import (
        is_windows,
        has_selector_event_loop_policy,
        is_playwright_imported,
        should_use_selector_loop,
        get_current_event_loop_policy,
        setup_event_loop,
        validate_event_loop_setup,
        print_event_loop_info,
    )
except ImportError:
    from event_loop_fix import (
        is_windows,
        has_selector_event_loop_policy,
        is_playwright_imported,
        should_use_selector_loop,
        get_current_event_loop_policy,
        setup_event_loop,
        validate_event_loop_setup,
        print_event_loop_info,
    )


class TestEventLoopFixPlatformDetection:
    """Testes para detecção de plataforma."""

    def test_is_windows_true(self):
        """Testa detecção de Windows quando é Windows."""
        with patch("platform.system", return_value="Windows"):
            result = is_windows()
            assert result is True

    def test_is_windows_false(self):
        """Testa detecção de Windows quando não é Windows."""
        with patch("platform.system", return_value="Linux"):
            result = is_windows()
            assert result is False

    def test_is_windows_case_insensitive(self):
        """Testa detecção de Windows com diferentes casos."""
        with patch("platform.system", return_value="windows"):
            result = is_windows()
            assert result is True

    def test_is_windows_mixed_case(self):
        """Testa detecção de Windows com caso misto."""
        with patch("platform.system", return_value="WinDoWs"):
            result = is_windows()
            assert result is True


class TestEventLoopFixPolicyDetection:
    """Testes para detecção de políticas de event loop."""

    def test_has_selector_event_loop_policy_windows_true(self):
        """Testa detecção de política selector no Windows quando disponível."""
        with patch("src.event_loop_fix.is_windows", return_value=True):
            with patch("asyncio.WindowsSelectorEventLoopPolicy", create=True):
                result = has_selector_event_loop_policy()
                assert result is True

    def test_has_selector_event_loop_policy_windows_false(self):
        """Testa detecção de política selector no Windows quando não disponível."""
        with patch("src.event_loop_fix.is_windows", return_value=True):
            with patch("asyncio.WindowsSelectorEventLoopPolicy", None):
                result = has_selector_event_loop_policy()
                assert result is False

    def test_has_selector_event_loop_policy_not_windows(self):
        """Testa detecção de política selector quando não é Windows."""
        with patch("src.event_loop_fix.is_windows", return_value=False):
            result = has_selector_event_loop_policy()
            assert result is False


class TestEventLoopFixPlaywrightDetection:
    """Testes para detecção de Playwright."""

    def test_is_playwright_imported_true_playwright(self):
        """Testa detecção quando playwright está importado."""
        with patch.dict(sys.modules, {"playwright": Mock()}):
            result = is_playwright_imported()
            assert result is True

    def test_is_playwright_imported_true_crawl4ai(self):
        """Testa detecção quando crawl4ai está importado."""
        with patch.dict(sys.modules, {"crawl4ai": Mock()}):
            result = is_playwright_imported()
            assert result is True

    def test_is_playwright_imported_false(self):
        """Testa detecção quando nem playwright nem crawl4ai estão importados."""
        # Remove as chaves se existirem
        sys_modules_copy = sys.modules.copy()
        if "playwright" in sys_modules_copy:
            del sys_modules_copy["playwright"]
        if "crawl4ai" in sys_modules_copy:
            del sys_modules_copy["crawl4ai"]

        with patch.dict(sys.modules, sys_modules_copy, clear=True):
            result = is_playwright_imported()
            assert result is False

    def test_is_playwright_imported_both_modules(self):
        """Testa detecção quando ambos os módulos estão importados."""
        with patch.dict(sys.modules, {"playwright": Mock(), "crawl4ai": Mock()}):
            result = is_playwright_imported()
            assert result is True


class TestEventLoopFixSelectorLoopDecision:
    """Testes para decisão de uso do selector loop."""

    def test_should_use_selector_loop_windows_available_no_playwright(self):
        """Testa decisão de usar selector loop no Windows sem Playwright."""
        with patch(
            "src.event_loop_fix.has_selector_event_loop_policy", return_value=True
        ):
            with patch("src.event_loop_fix.is_playwright_imported", return_value=False):
                result = should_use_selector_loop()
                assert result is True

    def test_should_use_selector_loop_windows_available_with_playwright(self):
        """Testa decisão de não usar selector loop no Windows com Playwright."""
        with patch(
            "src.event_loop_fix.has_selector_event_loop_policy", return_value=True
        ):
            with patch("src.event_loop_fix.is_playwright_imported", return_value=True):
                result = should_use_selector_loop()
                assert result is False

    def test_should_use_selector_loop_windows_unavailable(self):
        """Testa decisão de não usar selector loop quando não disponível no Windows."""
        with patch(
            "src.event_loop_fix.has_selector_event_loop_policy", return_value=False
        ):
            result = should_use_selector_loop()
            assert result is False


class TestEventLoopFixCurrentPolicy:
    """Testes para obtenção da política atual de event loop."""

    def test_get_current_event_loop_policy_success(self):
        """Testa obtenção bem-sucedida da política atual."""
        mock_policy = Mock()
        mock_policy.__class__.__name__ = "WindowsProactorEventLoopPolicy"

        with patch("asyncio.get_event_loop_policy", return_value=mock_policy):
            result = get_current_event_loop_policy()
            assert result == "WindowsProactorEventLoopPolicy"

    def test_get_current_event_loop_policy_error(self):
        """Testa obtenção da política atual com erro."""
        with patch("asyncio.get_event_loop_policy", side_effect=Exception("Erro")):
            result = get_current_event_loop_policy()
            assert result == "Unknown"


class TestEventLoopFixSetup:
    """Testes para configuração de event loop."""

    def test_setup_event_loop_windows_selector_available(self):
        """Testa configuração de event loop no Windows com selector disponível."""
        with patch("src.event_loop_fix.is_windows", return_value=True):
            with patch(
                "src.event_loop_fix.has_selector_event_loop_policy", return_value=True
            ):
                with patch(
                    "src.event_loop_fix.is_playwright_imported", return_value=False
                ):
                    with patch(
                        "asyncio.WindowsSelectorEventLoopPolicy"
                    ) as mock_policy_class:
                        mock_policy = Mock()
                        mock_policy_class.return_value = mock_policy

                        with patch("asyncio.set_event_loop_policy") as mock_set_policy:
                            result = setup_event_loop()

                            assert result == "WindowsSelectorEventLoopPolicy"
                            mock_policy_class.assert_called_once()
                            mock_set_policy.assert_called_once_with(mock_policy)

    def test_setup_event_loop_windows_playwright_imported(self):
        """Testa configuração de event loop no Windows com Playwright importado."""
        with patch("src.event_loop_fix.is_windows", return_value=True):
            with patch(
                "src.event_loop_fix.has_selector_event_loop_policy", return_value=True
            ):
                with patch(
                    "src.event_loop_fix.is_playwright_imported", return_value=True
                ):
                    with patch(
                        "asyncio.WindowsProactorEventLoopPolicy"
                    ) as mock_policy_class:
                        mock_policy = Mock()
                        mock_policy_class.return_value = mock_policy

                        with patch("asyncio.set_event_loop_policy") as mock_set_policy:
                            result = setup_event_loop()

                            assert result == "WindowsProactorEventLoopPolicy"
                            mock_policy_class.assert_called_once()
                            mock_set_policy.assert_called_once_with(mock_policy)

    def test_setup_event_loop_windows_no_selector_available(self):
        """Testa configuração de event loop no Windows sem selector disponível."""
        with patch("src.event_loop_fix.is_windows", return_value=True):
            with patch(
                "src.event_loop_fix.has_selector_event_loop_policy", return_value=False
            ):
                with patch(
                    "asyncio.WindowsProactorEventLoopPolicy"
                ) as mock_policy_class:
                    mock_policy = Mock()
                    mock_policy_class.return_value = mock_policy

                    with patch("asyncio.set_event_loop_policy") as mock_set_policy:
                        result = setup_event_loop()

                        assert result == "WindowsProactorEventLoopPolicy"
                        mock_policy_class.assert_called_once()
                        mock_set_policy.assert_called_once_with(mock_policy)

    def test_setup_event_loop_not_windows(self):
        """Testa configuração de event loop quando não é Windows."""
        with patch("src.event_loop_fix.is_windows", return_value=False):
            result = setup_event_loop()
            assert result is None

    def test_setup_event_loop_error_handling(self):
        """Testa tratamento de erro na configuração de event loop."""
        with patch("src.event_loop_fix.is_windows", return_value=True):
            with patch(
                "src.event_loop_fix.has_selector_event_loop_policy", return_value=True
            ):
                with patch(
                    "src.event_loop_fix.is_playwright_imported", return_value=False
                ):
                    with patch(
                        "asyncio.WindowsSelectorEventLoopPolicy",
                        side_effect=Exception("Erro"),
                    ):
                        result = setup_event_loop()
                        assert result is None


class TestEventLoopFixValidation:
    """Testes para validação de configuração de event loop."""

    def test_validate_event_loop_setup_windows_selector(self):
        """Testa validação de configuração no Windows com selector."""
        with patch("src.event_loop_fix.is_windows", return_value=True):
            with patch(
                "src.event_loop_fix.has_selector_event_loop_policy", return_value=True
            ):
                with patch(
                    "src.event_loop_fix.is_playwright_imported", return_value=False
                ):
                    with patch(
                        "src.event_loop_fix.get_current_event_loop_policy",
                        return_value="WindowsSelectorEventLoopPolicy",
                    ):
                        result = validate_event_loop_setup()

                        assert isinstance(result, dict)
                        assert "platform" in result
                        assert "windows" in result["platform"].lower()
                        assert "policy" in result
                        assert "selector" in result["policy"].lower()
                        assert "playwright_detected" in result
                        assert result["playwright_detected"] is False

    def test_validate_event_loop_setup_windows_proactor(self):
        """Testa validação de configuração no Windows com proactor."""
        with patch("src.event_loop_fix.is_windows", return_value=True):
            with patch(
                "src.event_loop_fix.has_selector_event_loop_policy", return_value=True
            ):
                with patch(
                    "src.event_loop_fix.is_playwright_imported", return_value=True
                ):
                    with patch(
                        "src.event_loop_fix.get_current_event_loop_policy",
                        return_value="WindowsProactorEventLoopPolicy",
                    ):
                        result = validate_event_loop_setup()

                        assert isinstance(result, dict)
                        assert "platform" in result
                        assert "windows" in result["platform"].lower()
                        assert "policy" in result
                        assert "proactor" in result["policy"].lower()
                        assert "playwright_detected" in result
                        assert result["playwright_detected"] is True

    def test_validate_event_loop_setup_not_windows(self):
        """Testa validação de configuração quando não é Windows."""
        with patch("src.event_loop_fix.is_windows", return_value=False):
            with patch(
                "src.event_loop_fix.get_current_event_loop_policy",
                return_value="DefaultEventLoopPolicy",
            ):
                result = validate_event_loop_setup()

                assert isinstance(result, dict)
                assert "platform" in result
                assert "windows" not in result["platform"].lower()
                assert "policy" in result
                assert "playwright_detected" in result

    def test_validate_event_loop_setup_error_handling(self):
        """Testa tratamento de erro na validação."""
        with patch("src.event_loop_fix.is_windows", side_effect=Exception("Erro")):
            result = validate_event_loop_setup()

            assert isinstance(result, dict)
            assert "error" in result


class TestEventLoopFixPrintInfo:
    """Testes para impressão de informações de event loop."""

    def test_print_event_loop_info(self):
        """Testa impressão de informações de event loop."""
        mock_validation_result = {
            "platform": "Windows",
            "policy": "WindowsSelectorEventLoopPolicy",
            "playwright_detected": False,
        }

        with patch(
            "src.event_loop_fix.validate_event_loop_setup",
            return_value=mock_validation_result,
        ):
            # Não deve gerar erro
            print_event_loop_info()


class TestEventLoopFixIntegration:
    """Testes de integração para event_loop_fix."""

    def test_complete_event_loop_workflow_windows_selector(self):
        """Testa workflow completo de event loop no Windows com selector."""
        with patch("src.event_loop_fix.is_windows", return_value=True):
            with patch(
                "src.event_loop_fix.has_selector_event_loop_policy", return_value=True
            ):
                with patch(
                    "src.event_loop_fix.is_playwright_imported", return_value=False
                ):
                    with patch(
                        "asyncio.WindowsSelectorEventLoopPolicy"
                    ) as mock_policy_class:
                        mock_policy = Mock()
                        mock_policy_class.return_value = mock_policy

                        with patch("asyncio.set_event_loop_policy") as mock_set_policy:
                            with patch(
                                "src.event_loop_fix.get_current_event_loop_policy",
                                return_value="WindowsSelectorEventLoopPolicy",
                            ):
                                # Testa configuração
                                setup_result = setup_event_loop()
                                assert setup_result == "WindowsSelectorEventLoopPolicy"

                                # Testa validação
                                validation_result = validate_event_loop_setup()
                                assert (
                                    validation_result["policy"]
                                    == "WindowsSelectorEventLoopPolicy"
                                )
                                assert validation_result["playwright_detected"] is False

    def test_complete_event_loop_workflow_windows_proactor(self):
        """Testa workflow completo de event loop no Windows com proactor."""
        with patch("src.event_loop_fix.is_windows", return_value=True):
            with patch(
                "src.event_loop_fix.has_selector_event_loop_policy", return_value=True
            ):
                with patch(
                    "src.event_loop_fix.is_playwright_imported", return_value=True
                ):
                    with patch(
                        "asyncio.WindowsProactorEventLoopPolicy"
                    ) as mock_policy_class:
                        mock_policy = Mock()
                        mock_policy_class.return_value = mock_policy

                        with patch("asyncio.set_event_loop_policy") as mock_set_policy:
                            with patch(
                                "src.event_loop_fix.get_current_event_loop_policy",
                                return_value="WindowsProactorEventLoopPolicy",
                            ):
                                # Testa configuração
                                setup_result = setup_event_loop()
                                assert setup_result == "WindowsProactorEventLoopPolicy"

                                # Testa validação
                                validation_result = validate_event_loop_setup()
                                assert (
                                    validation_result["policy"]
                                    == "WindowsProactorEventLoopPolicy"
                                )
                                assert validation_result["playwright_detected"] is True

    def test_complete_event_loop_workflow_not_windows(self):
        """Testa workflow completo de event loop quando não é Windows."""
        with patch("src.event_loop_fix.is_windows", return_value=False):
            with patch(
                "src.event_loop_fix.get_current_event_loop_policy",
                return_value="DefaultEventLoopPolicy",
            ):
                # Testa configuração
                setup_result = setup_event_loop()
                assert setup_result is None

                # Testa validação
                validation_result = validate_event_loop_setup()
                assert "platform" in validation_result
                assert "windows" not in validation_result["platform"].lower()

    def test_event_loop_decision_logic(self):
        """Testa lógica de decisão de event loop."""
        test_cases = [
            # (is_windows, has_selector, is_playwright, expected_selector)
            (True, True, False, True),  # Windows, selector disponível, sem Playwright
            (True, True, True, False),  # Windows, selector disponível, com Playwright
            (
                True,
                False,
                False,
                False,
            ),  # Windows, selector indisponível, sem Playwright
            (
                True,
                False,
                True,
                False,
            ),  # Windows, selector indisponível, com Playwright
            (False, True, False, False),  # Não Windows, selector disponível
            (
                False,
                True,
                True,
                False,
            ),  # Não Windows, selector disponível, com Playwright
        ]

        for is_win, has_sel, is_play, expected in test_cases:
            with patch("src.event_loop_fix.is_windows", return_value=is_win):
                with patch(
                    "src.event_loop_fix.has_selector_event_loop_policy",
                    return_value=has_sel,
                ):
                    with patch(
                        "src.event_loop_fix.is_playwright_imported",
                        return_value=is_play,
                    ):
                        result = should_use_selector_loop()
                        assert result == expected, (
                            f"Falhou para: Windows={is_win}, Selector={has_sel}, Playwright={is_play}"
                        )


class TestEventLoopFixEdgeCases:
    """Testes para casos extremos de event_loop_fix."""

    def test_setup_event_loop_policy_creation_error(self):
        """Testa erro na criação da política de event loop."""
        with patch("src.event_loop_fix.is_windows", return_value=True):
            with patch(
                "src.event_loop_fix.has_selector_event_loop_policy", return_value=True
            ):
                with patch(
                    "src.event_loop_fix.is_playwright_imported", return_value=False
                ):
                    with patch(
                        "asyncio.WindowsSelectorEventLoopPolicy",
                        side_effect=Exception("Erro de criação"),
                    ):
                        result = setup_event_loop()
                        assert result is None

    def test_setup_event_loop_policy_set_error(self):
        """Testa erro na configuração da política de event loop."""
        with patch("src.event_loop_fix.is_windows", return_value=True):
            with patch(
                "src.event_loop_fix.has_selector_event_loop_policy", return_value=True
            ):
                with patch(
                    "src.event_loop_fix.is_playwright_imported", return_value=False
                ):
                    with patch(
                        "asyncio.WindowsSelectorEventLoopPolicy"
                    ) as mock_policy_class:
                        mock_policy = Mock()
                        mock_policy_class.return_value = mock_policy

                        with patch(
                            "asyncio.set_event_loop_policy",
                            side_effect=Exception("Erro de configuração"),
                        ):
                            result = setup_event_loop()
                            assert result is None

    def test_validate_event_loop_setup_missing_attributes(self):
        """Testa validação com atributos ausentes."""
        with patch("src.event_loop_fix.is_windows", return_value=True):
            with patch(
                "src.event_loop_fix.has_selector_event_loop_policy", return_value=True
            ):
                with patch(
                    "src.event_loop_fix.is_playwright_imported", return_value=False
                ):
                    with patch(
                        "src.event_loop_fix.get_current_event_loop_policy",
                        side_effect=Exception("Erro"),
                    ):
                        result = validate_event_loop_setup()
                        assert isinstance(result, dict)
                        assert "error" in result

    def test_platform_detection_edge_cases(self):
        """Testa casos extremos de detecção de plataforma."""
        edge_cases = [
            "Windows",
            "windows",
            "WINDOWS",
            "WinDoWs",
            "Linux",
            "Darwin",
            "FreeBSD",
            "",
            None,
        ]

        for platform_name in edge_cases:
            with patch("platform.system", return_value=platform_name):
                result = is_windows()
                if platform_name and platform_name.lower() == "windows":
                    assert result is True
                else:
                    assert result is False

    def test_playwright_detection_edge_cases(self):
        """Testa casos extremos de detecção de Playwright."""
        # Testa com módulos vazios
        with patch.dict(sys.modules, {}, clear=True):
            result = is_playwright_imported()
            assert result is False

        # Testa com módulos None
        with patch.dict(sys.modules, {"playwright": None, "crawl4ai": None}):
            result = is_playwright_imported()
            assert result is True  # Ainda detecta como importado mesmo que None

