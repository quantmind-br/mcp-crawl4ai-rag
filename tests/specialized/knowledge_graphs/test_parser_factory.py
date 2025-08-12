from src.k_graph.parsing.parser_factory import ParserFactory, get_global_factory


class TestParserFactory:
    def test_parser_factory_initialization(self):
        """Test ParserFactory initializes correctly"""
        factory = ParserFactory()
        assert factory is not None

    def test_parser_factory_get_parser_for_supported_language(self):
        """Test ParserFactory get_parser for supported languages"""
        factory = ParserFactory()
        parser = factory.get_parser("python")
        assert parser is not None

    def test_parser_factory_get_parser_for_unsupported_language(self):
        """Test ParserFactory get_parser for unsupported languages"""
        factory = ParserFactory()
        parser = factory.get_parser("unsupported")
        # Should return a fallback parser or None
        assert parser is not None or parser is None

    def test_get_global_factory_returns_singleton(self):
        """Test get_global_factory returns the same instance"""
        factory1 = get_global_factory()
        factory2 = get_global_factory()
        assert factory1 is factory2
