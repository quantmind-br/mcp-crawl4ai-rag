#!/usr/bin/env python3
"""
Integration tests for multi-language Tree-sitter integration.

Tests the complete integration flow from Tree-sitter parsing through
Neo4j population and AI script analysis across multiple languages.
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path
import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from knowledge_graphs.parse_repo_into_neo4j import Neo4jCodeAnalyzer
    from knowledge_graphs.ai_script_analyzer import AIScriptAnalyzer
    from knowledge_graphs.parser_factory import get_global_factory

    _TS_AVAILABLE = True
except Exception:
    Neo4jCodeAnalyzer = None
    AIScriptAnalyzer = None

    def get_global_factory():
        return None

    _TS_AVAILABLE = False

# Runtime check for tree_sitter availability
try:
    _TS_RUNTIME = True
except Exception:
    _TS_RUNTIME = False

pytestmark = pytest.mark.skipif(
    not (_TS_AVAILABLE and _TS_RUNTIME), reason="Tree-sitter not available in env"
)


class TestMultiLanguageIntegration(unittest.TestCase):
    """Test complete multi-language integration workflow."""

    def setUp(self):
        """Set up test fixtures."""
        if not (_TS_AVAILABLE and _TS_RUNTIME):
            self.skipTest("Tree-sitter not available")
        self.neo4j_analyzer = Neo4jCodeAnalyzer()
        self.ai_analyzer = AIScriptAnalyzer()
        self.parser_factory = get_global_factory()

        # Test code samples for different languages
        self.test_code_samples = {
            "python_sample.py": '''
import asyncio
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Config:
    name: str
    value: int = 100

class DataProcessor:
    """A sample data processing class."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data = []
    
    async def process_async(self, items: List[str]) -> List[str]:
        """Process items asynchronously."""
        results = []
        for item in items:
            processed = await self._process_item(item)
            results.append(processed)
        return results
    
    async def _process_item(self, item: str) -> str:
        await asyncio.sleep(0.01)  # Simulate async work
        return f"processed_{item}"
    
    def get_stats(self) -> dict:
        return {
            "config_name": self.config.name,
            "data_count": len(self.data)
        }

def create_processor(name: str = "default") -> DataProcessor:
    config = Config(name=name)
    return DataProcessor(config)

# Usage example
async def main():
    processor = create_processor("test")
    items = ["item1", "item2", "item3"]
    results = await processor.process_async(items)
    stats = processor.get_stats()
    print(f"Processed {len(results)} items")

if __name__ == "__main__":
    asyncio.run(main())
''',
            "javascript_sample.js": """
const fs = require('fs');
const path = require('path');

class ConfigManager {
    constructor(configPath) {
        this.configPath = configPath;
        this.config = {};
        this.loadConfig();
    }
    
    loadConfig() {
        try {
            const data = fs.readFileSync(this.configPath, 'utf8');
            this.config = JSON.parse(data);
        } catch (error) {
            console.error('Failed to load config:', error.message);
            this.config = this.getDefaultConfig();
        }
    }
    
    getDefaultConfig() {
        return {
            name: 'default',
            value: 100,
            enabled: true
        };
    }
    
    saveConfig() {
        const data = JSON.stringify(this.config, null, 2);
        fs.writeFileSync(this.configPath, data);
    }
    
    get(key) {
        return this.config[key];
    }
    
    set(key, value) {
        this.config[key] = value;
        this.saveConfig();
    }
}

function createConfigManager(configPath) {
    return new ConfigManager(configPath);
}

async function processItems(items, processor) {
    const results = [];
    for (const item of items) {
        const result = await processor(item);
        results.push(result);
    }
    return results;
}

function validateConfig(config) {
    const required = ['name', 'value'];
    return required.every(key => key in config);
}

// Usage
const configManager = createConfigManager('./config.json');
configManager.set('debug', true);

module.exports = {
    ConfigManager,
    createConfigManager,
    processItems,
    validateConfig
};
""",
            "JavaSample.java": """
package com.example.sample;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.CompletableFuture;

public class JavaSample {
    
    public static class Config {
        private final String name;
        private final int value;
        
        public Config(String name, int value) {
            this.name = name;
            this.value = value;
        }
        
        public String getName() { return name; }
        public int getValue() { return value; }
    }
    
    public static class DataService {
        private final Config config;
        private final List<String> data;
        
        public DataService(Config config) {
            this.config = config;
            this.data = new ArrayList<>();
        }
        
        public CompletableFuture<List<String>> processAsync(List<String> items) {
            return CompletableFuture.supplyAsync(() -> {
                List<String> results = new ArrayList<>();
                for (String item : items) {
                    results.add("processed_" + item);
                }
                return results;
            });
        }
        
        public void addData(String item) {
            data.add(item);
        }
        
        public Map<String, Object> getStats() {
            Map<String, Object> stats = new HashMap<>();
            stats.put("configName", config.getName());
            stats.put("dataCount", data.size());
            return stats;
        }
    }
    
    public static DataService createService(String name) {
        Config config = new Config(name, 100);
        return new DataService(config);
    }
    
    public static boolean validateData(List<String> data) {
        return data != null && !data.isEmpty();
    }
    
    public static void main(String[] args) {
        DataService service = createService("test");
        service.addData("sample");
        
        List<String> items = List.of("item1", "item2");
        service.processAsync(items)
            .thenAccept(results -> {
                System.out.println("Processed " + results.size() + " items");
            });
    }
}
""",
            "go_sample.go": """
package main

import (
    "fmt"
    "sync"
    "time"
)

type Config struct {
    Name  string
    Value int
}

func NewConfig(name string, value int) *Config {
    return &Config{
        Name:  name,
        Value: value,
    }
}

type DataProcessor struct {
    config *Config
    data   []string
    mutex  sync.RWMutex
}

func NewDataProcessor(config *Config) *DataProcessor {
    return &DataProcessor{
        config: config,
        data:   make([]string, 0),
    }
}

func (dp *DataProcessor) ProcessAsync(items []string) chan string {
    results := make(chan string, len(items))
    
    go func() {
        defer close(results)
        var wg sync.WaitGroup
        
        for _, item := range items {
            wg.Add(1)
            go func(item string) {
                defer wg.Done()
                time.Sleep(10 * time.Millisecond) // Simulate work
                results <- fmt.Sprintf("processed_%s", item)
            }(item)
        }
        
        wg.Wait()
    }()
    
    return results
}

func (dp *DataProcessor) AddData(item string) {
    dp.mutex.Lock()
    defer dp.mutex.Unlock()
    dp.data = append(dp.data, item)
}

func (dp *DataProcessor) GetStats() map[string]interface{} {
    dp.mutex.RLock()
    defer dp.mutex.RUnlock()
    
    return map[string]interface{}{
        "configName": dp.config.Name,
        "dataCount":  len(dp.data),
    }
}

func CreateProcessor(name string) *DataProcessor {
    config := NewConfig(name, 100)
    return NewDataProcessor(config)
}

func ValidateItems(items []string) bool {
    return len(items) > 0
}

func main() {
    processor := CreateProcessor("test")
    processor.AddData("sample")
    
    items := []string{"item1", "item2", "item3"}
    results := processor.ProcessAsync(items)
    
    for result := range results {
        fmt.Printf("Result: %s\\n", result)
    }
    
    stats := processor.GetStats()
    fmt.Printf("Stats: %+v\\n", stats)
}
""",
        }

    def test_neo4j_multi_language_analysis(self):
        """Test Neo4j analyzer handles multiple languages correctly."""
        results = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for filename, content in self.test_code_samples.items():
                file_path = temp_path / filename
                file_path.write_text(content)

                # Analyze file with Neo4j analyzer
                try:
                    analysis = self.neo4j_analyzer.analyze_file(
                        file_path, temp_path, {"test"}
                    )
                    results[filename] = analysis

                    # Validate analysis structure
                    self.assertIsInstance(analysis, dict)
                    self.assertIn("language", analysis)
                    self.assertIn("classes", analysis)
                    self.assertIn("functions", analysis)
                    self.assertIn("imports", analysis)

                    # Check language detection
                    expected_languages = {
                        "python_sample.py": "python",
                        "javascript_sample.js": "javascript",
                        "JavaSample.java": "java",
                        "go_sample.go": "go",
                    }
                    expected_lang = expected_languages.get(filename)
                    if expected_lang:
                        self.assertEqual(
                            analysis["language"],
                            expected_lang,
                            f"Wrong language detected for {filename}",
                        )

                    # Validate content extraction based on language expectations
                    if filename.endswith(".java"):
                        # Java typically has classes with methods, not standalone functions
                        self.assertGreater(
                            len(analysis["classes"]),
                            0,
                            f"No classes found in {filename}",
                        )
                    elif filename.endswith(".go"):
                        # Go has structs (mapped to classes) and functions
                        self.assertGreater(
                            len(analysis["classes"]) + len(analysis["functions"]),
                            0,
                            f"No structs/interfaces/functions found in {filename}",
                        )
                    else:
                        # Other languages should have both
                        self.assertGreater(
                            len(analysis["classes"]),
                            0,
                            f"No classes found in {filename}",
                        )
                        self.assertGreater(
                            len(analysis["functions"]),
                            0,
                            f"No functions found in {filename}",
                        )

                except Exception as e:
                    self.fail(f"Neo4j analysis failed for {filename}: {e}")

        # Validate all files were processed
        self.assertEqual(len(results), len(self.test_code_samples))

    def test_ai_script_multi_language_analysis(self):
        """Test AI script analyzer handles multiple languages correctly."""
        results = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for filename, content in self.test_code_samples.items():
                file_path = temp_path / filename
                file_path.write_text(content)

                # Analyze file with AI script analyzer
                try:
                    analysis = self.ai_analyzer.analyze_script(str(file_path))
                    results[filename] = analysis

                    # Validate analysis structure
                    self.assertIsNotNone(analysis)
                    self.assertEqual(analysis.file_path, str(file_path))

                    # Check language detection
                    expected_languages = {
                        "python_sample.py": "python",
                        "javascript_sample.js": "javascript",
                        "JavaSample.java": "java",
                        "go_sample.go": "go",
                    }
                    expected_lang = expected_languages.get(filename)
                    if expected_lang:
                        self.assertEqual(
                            analysis.language,
                            expected_lang,
                            f"Wrong language detected for {filename}",
                        )

                    # Validate Tree-sitter data is present
                    self.assertIsNotNone(
                        analysis.tree_sitter_data, f"No Tree-sitter data for {filename}"
                    )

                    # Check basic extraction worked based on language expectations
                    ts_data = analysis.tree_sitter_data
                    if filename.endswith(".java"):
                        # Java typically has classes with methods, not standalone functions
                        self.assertGreater(
                            len(ts_data.classes), 0, f"No classes found in {filename}"
                        )
                    elif filename.endswith(".go"):
                        # Go has structs (mapped to classes) and/or functions
                        self.assertGreater(
                            len(ts_data.classes) + len(ts_data.functions),
                            0,
                            f"No structs/interfaces/functions found in {filename}",
                        )
                    else:
                        # Other languages should have classes
                        self.assertGreater(
                            len(ts_data.classes), 0, f"No classes found in {filename}"
                        )

                    # For Python, also check detailed AST analysis
                    if filename.endswith(".py"):
                        self.assertGreater(
                            len(analysis.imports), 0, "No imports found in Python file"
                        )
                        self.assertGreater(
                            len(analysis.class_instantiations),
                            0,
                            "No class instantiations found",
                        )

                except Exception as e:
                    self.fail(f"AI script analysis failed for {filename}: {e}")

        # Validate all files were processed
        self.assertEqual(len(results), len(self.test_code_samples))

    def test_cross_validation_consistency(self):
        """Test that Tree-sitter and AST analysis produce consistent results for Python."""
        python_code = self.test_code_samples["python_sample.py"]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file_path = temp_path / "test.py"
            file_path.write_text(python_code)

            # Get both analyses
            neo4j_analysis = self.neo4j_analyzer.analyze_file(
                file_path, temp_path, {"test"}
            )
            ai_analysis = self.ai_analyzer.analyze_script(str(file_path))

            # Both should detect Python
            self.assertEqual(neo4j_analysis["language"], "python")
            self.assertEqual(ai_analysis.language, "python")

            # Class counts should be consistent
            neo4j_classes = {cls["name"] for cls in neo4j_analysis["classes"]}
            ts_classes = {cls["name"] for cls in ai_analysis.tree_sitter_data.classes}

            # Should have substantial overlap (allowing for minor differences in detection)
            common_classes = neo4j_classes.intersection(ts_classes)
            self.assertGreater(
                len(common_classes), 0, "No common classes detected between analyzers"
            )

            # Function counts should be reasonable
            self.assertGreater(len(neo4j_analysis["functions"]), 0)
            self.assertGreater(len(ai_analysis.tree_sitter_data.functions), 0)

    def test_parser_factory_integration(self):
        """Test parser factory integration across the system."""
        # Test that all components use the same factory instance
        factory1 = get_global_factory()
        factory2 = self.parser_factory
        factory3 = self.neo4j_analyzer.parser_factory

        # Should all reference the same singleton
        self.assertIs(factory1, factory2)
        self.assertIs(factory2, factory3)

        # Test cache consistency
        languages_to_test = ["python", "javascript", "java"]
        for language in languages_to_test:
            parser1 = factory1.get_parser(language)
            parser2 = factory2.get_parser(language)
            parser3 = factory3.get_parser(language)

            # Should return same cached instances
            self.assertIs(parser1, parser2)
            self.assertIs(parser2, parser3)

    def test_error_handling_across_languages(self):
        """Test error handling for problematic code across languages."""
        problematic_samples = {
            "broken_python.py": """
class BrokenClass:
    def __init__(self
        # Missing closing parenthesis
        pass
    
    def method(self):
        return undefined_variable
""",
            "broken_js.js": """
class BrokenClass {
    constructor() {
        this.value = undefined_function();
    }
    
    method( {
        // Missing parameter closing
        return this.value;
    }
}
""",
            "Broken.java": """
public class Broken {
    public Broken() {
        // Missing semicolon
        int x = 5
    }
    
    public void method() {
        undefinedMethod();
    }
}
""",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for filename, content in problematic_samples.items():
                file_path = temp_path / filename
                file_path.write_text(content)

                # Both analyzers should handle errors gracefully
                try:
                    neo4j_result = self.neo4j_analyzer.analyze_file(
                        file_path, temp_path, {"test"}
                    )
                    # Should return some result, even if partial
                    self.assertIsInstance(neo4j_result, dict)

                except Exception as e:
                    self.fail(f"Neo4j analyzer crashed on {filename}: {e}")

                try:
                    ai_result = self.ai_analyzer.analyze_script(str(file_path))
                    # Should return analysis result, possibly with errors
                    self.assertIsNotNone(ai_result)

                except Exception as e:
                    self.fail(f"AI analyzer crashed on {filename}: {e}")

    def test_performance_across_languages(self):
        """Test parsing performance across different languages."""
        import time

        performance_results = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for filename, content in self.test_code_samples.items():
                file_path = temp_path / filename
                file_path.write_text(content)

                # Time the parsing
                start_time = time.time()
                analysis = self.ai_analyzer.analyze_script(str(file_path))
                parse_time = time.time() - start_time

                performance_results[filename] = {
                    "parse_time": parse_time,
                    "language": analysis.language,
                    "classes": len(analysis.tree_sitter_data.classes),
                    "functions": len(analysis.tree_sitter_data.functions),
                }

                # Performance should be reasonable (< 1 second per file)
                self.assertLess(
                    parse_time,
                    1.0,
                    f"Parsing {filename} took too long: {parse_time:.2f}s",
                )

        # All languages should parse in reasonable time
        total_time = sum(r["parse_time"] for r in performance_results.values())
        self.assertLess(
            total_time, 5.0, f"Total parsing time too long: {total_time:.2f}s"
        )


class TestLanguageSpecificFeatures(unittest.TestCase):
    """Test language-specific parsing features."""

    def setUp(self):
        """Set up test fixtures."""
        if not (_TS_AVAILABLE and _TS_RUNTIME):
            self.skipTest("Tree-sitter not available")
        self.factory = get_global_factory()

    def test_python_async_await_detection(self):
        """Test Python async/await pattern detection."""
        python_async_code = """
import asyncio

async def async_function():
    result = await some_async_call()
    return result

class AsyncClass:
    async def async_method(self):
        async with some_context() as ctx:
            await ctx.process()
"""

        parser = self.factory.get_parser("python")
        result = parser.parse(python_async_code, "async_test.py")

        # Should detect async functions
        function_names = {func["name"] for func in result.functions}
        self.assertIn("async_function", function_names)

        # Should detect classes with async methods
        class_names = {cls["name"] for cls in result.classes}
        self.assertIn("AsyncClass", class_names)

    def test_javascript_es6_features(self):
        """Test JavaScript ES6+ feature detection."""
        js_es6_code = """
import { Component } from 'react';
export default class ES6Class extends Component {
    constructor(props) {
        super(props);
        this.state = { value: 0 };
    }
    
    handleClick = () => {
        this.setState({ value: this.state.value + 1 });
    }
    
    render() {
        return <div onClick={this.handleClick}>{this.state.value}</div>;
    }
}

const ArrowFunction = (x, y) => x + y;

function* GeneratorFunction() {
    yield 1;
    yield 2;
}
"""

        parser = self.factory.get_parser("javascript")
        result = parser.parse(js_es6_code, "es6_test.js")

        # Should detect ES6 class
        class_names = {cls["name"] for cls in result.classes}
        self.assertIn("ES6Class", class_names)

        # Should detect various function types
        function_names = {func["name"] for func in result.functions}
        self.assertTrue(len(function_names) > 0)

    def test_java_generic_and_annotation_handling(self):
        """Test Java generics and annotation handling."""
        java_generic_code = """
import java.util.List;
import java.util.ArrayList;

@Entity
@Table(name = "users")
public class GenericService<T extends BaseEntity> {
    
    @Autowired
    private Repository<T> repository;
    
    public <R> List<R> processItems(List<T> items, Function<T, R> mapper) {
        List<R> results = new ArrayList<>();
        for (T item : items) {
            results.add(mapper.apply(item));
        }
        return results;
    }
    
    @Transactional
    public void saveAll(List<T> entities) {
        repository.saveAll(entities);
    }
}
"""

        parser = self.factory.get_parser("java")
        result = parser.parse(java_generic_code, "generic_test.java")

        # Should detect generic class
        class_names = {cls["name"] for cls in result.classes}
        self.assertIn("GenericService", class_names)

        # Should handle imports
        self.assertGreater(len(result.imports), 0)


if __name__ == "__main__":
    print("Running Multi-Language Integration Tests...")
    print("=" * 70)

    # Run tests with detailed output
    unittest.main(verbosity=2, exit=False)

    print("\n" + "=" * 70)
    print("Multi-Language Integration Tests Complete")
