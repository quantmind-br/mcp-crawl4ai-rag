"""
Performance benchmarks for Qdrant native hybrid search vs semantic-only search.

This module provides quantitative measurements of the performance improvements
achieved by the hybrid search implementation.
"""

import os
import time
import json
import statistics
from typing import Dict, List, Any

try:
    from src.clients.qdrant_client import QdrantClientWrapper
    from memory_profiler import memory_usage

    PERF_TESTS = True
except ImportError:
    print("Memory profiler not available, using simple timing")
    PERF_TESTS = True


class HybridSearchBenchmark:
    """Benchmark suite for hybrid search performance analysis."""

    def __init__(self, device="cpu"):
        self.device = device
        self.results = {}

    def setup_test_environment(self):
        """Set up test environment with collection."""
        # Ensure local Qdrant is available
        os.environ["USE_HYBRID_SEARCH"] = "true"
        self.wrapper = QdrantClientWrapper(device=self.device)

    def teardown_test_environment(self):
        """Clean up test resources."""
        pass

    def create_test_documents(self, count: int) -> List[Dict[str, Any]]:
        """Generate realistic test documents for benchmarking."""
        topics = [
            "machine learning python algorithms",
            "neural networks deep learning",
            "data science artificial intelligence",
            "programming languages software development",
            "cloud computing distributed systems",
        ]

        documents = []
        base_embedding = [0.1] * 1024

        for i in range(count):
            topic_idx = i % len(topics)
            doc = {
                "id": f"doc_{i:04d}",
                "content": f"Document {i} about {topics[topic_idx]} implementation details and best practices",
                "dense_vector": [x + (i * 0.001) for x in base_embedding],
                "sparse_vector": {
                    "indices": [(i + j * 10) % 500 for j in range(20)],
                    "values": [1.0 + j * 0.1 for j in range(20)],
                },
                "metadata": {
                    "topic": topics[topic_idx],
                    "length": 50 + i,
                    "source": f"source_{i % 3}",
                },
            }
            documents.append(doc)

        return documents

    def benchmark_collection_creation(
        self, collection_name: str, use_hybrid: bool
    ) -> Dict[str, float]:
        """Benchmark collection creation performance."""
        start_time = time.time()

        try:
            success = self.wrapper.create_collection_if_not_exists(
                collection_name, use_hybrid=use_hybrid
            )

            creation_time = time.time() - start_time

            return {
                "creation_time": creation_time,
                "success": success,
                "type": "hybrid" if use_hybrid else "legacy",
            }

        except Exception as e:
            print(f"Collection creation failed: {e}")
            return {"creation_time": -1, "success": False, "type": "error"}

    def benchmark_insertion_performance(
        self, collection_name: str, documents: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Benchmark document insertion performance."""

        # Batch preparation
        from qdrant_client.models import PointStruct, SparseVector

        start_time = time.time()

        points = []
        for doc in documents:
            if doc.get("sparse_vector"):
                point = PointStruct(
                    id=doc["id"],
                    vector={
                        "text-dense": doc["dense_vector"],
                        "text-sparse": SparseVector(
                            indices=doc["sparse_vector"]["indices"],
                            values=doc["sparse_vector"]["values"],
                        ),
                    },
                    payload=doc["metadata"],
                )
            else:
                point = PointStruct(
                    id=doc["id"], vector=doc["dense_vector"], payload=doc["metadata"]
                )
            points.append(point)

        # Batch insertion
        insertion_start = time.time()

        try:
            result = self.wrapper.client.upsert(
                collection_name=collection_name, points=points
            )

            insertion_time = time.time() - insertion_start
            total_time = time.time() - start_time

            return {
                "insertion_time": insertion_time,
                "total_time": total_time,
                "docs_per_second": len(documents) / insertion_time,
                "success": result.operation_id is not None,
            }

        except Exception:
            return {
                "insertion_time": -1,
                "total_time": -1,
                "docs_per_second": -1,
                "success": False,
            }

    def benchmark_search_performance(
        self, collection_name: str, query_vector: List[float], test_types: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark search performance."""

        results = {}

        for test_type in test_types:
            times = []

            if test_type == "semantic_only":
                # Legacy-style semantic search
                for _ in range(10):  # Multiple runs for accurate timing
                    start = time.time()
                    try:
                        self.wrapper.client.search(
                            collection_name=collection_name,
                            query_vector=query_vector,
                            limit=10,
                        )
                        times.append(time.time() - start)
                    except Exception as e:
                        print(f"Semantic-only search failed: {e}")
                        break

            elif test_type == "hybrid_dense":
                # Hybrid dense vector search
                for _ in range(10):
                    start = time.time()
                    try:
                        self.wrapper.client.search(
                            collection_name=collection_name,
                            query_vector=query_vector,
                            limit=10,
                            using="text-dense",
                        )
                        times.append(time.time() - start)
                    except Exception as e:
                        print(f"Hybrid dense search failed: {e}")
                        break

            elif test_type == "hybrid_sparse":
                # Hybrid sparse vector search
                from qdrant_client.models import SparseVector

                sparse_query = SparseVector(indices=[1, 2, 3], values=[1.0, 0.8, 1.2])

                for _ in range(10):
                    start = time.time()
                    try:
                        self.wrapper.client.search(
                            collection_name=collection_name,
                            query_vector=sparse_query,
                            limit=10,
                            using="text-sparse",
                        )
                        times.append(time.time() - start)
                    except Exception as e:
                        print(f"Hybrid sparse search failed: {e}")
                        break

            elif test_type == "hybrid_batch":
                # Hybrid batch search (true hybrid efficiency)
                from qdrant_client.models import SparseVector

                dense_queries = [query_vector] * 5
                sparse_queries = [
                    SparseVector(indices=[1, 2, 3], values=[1.0, 0.8, 1.2])
                ] * 5

                for _ in range(5):  # Fewer iterations for batch tests
                    start = time.time()
                    try:
                        # Simulate batch hybrid search
                        self.wrapper.client.search_batch(
                            collection_name=collection_name,
                            requests=[
                                {
                                    "name": "text-dense",
                                    "vector": dense_queries[0],
                                    "limit": 10,
                                },
                                {
                                    "name": "text-sparse",
                                    "vector": sparse_queries[0],
                                    "limit": 10,
                                },
                            ],
                        )
                        times.append(time.time() - start)
                    except Exception as e:
                        print(f"Hybrid batch search failed: {e}")
                        break

            if times:
                results[test_type] = {
                    "min_time": min(times),
                    "max_time": max(times),
                    "avg_time": statistics.mean(times),
                    "median_time": statistics.median(times),
                }
            else:
                results[test_type] = {"error": "Test failed"}

        return results

    def run_comprehensive_benchmark(
        self, document_counts: List[int] = None
    ) -> Dict[str, Any]:
        """Run complete benchmark suite and generate report."""

        if document_counts is None:
            document_counts = [100, 500, 1000, 5000]

        print("🚀 Starting Qdrant Hybrid Search Performance Benchmark")
        print("=" * 60)

        report = {
            "metadata": {
                "device": self.device,
                "timestamp": int(time.time()),
                "document_counts": document_counts,
            },
            "results": {},
        }

        for count in document_counts:
            print(f"\n📊 Testing with {count:,} documents...")

            collection_name = f"benchmark_{count}"

            try:
                # Clean previous collection
                try:
                    self.wrapper.client.delete_collection(collection_name)
                except:
                    pass

                # Set up test environment
                documents = self.create_test_documents(count)
                query_vector = [0.12] * 1024

                # Create collections for comparison
                hybrid_col = f"{collection_name}_hybrid"
                legacy_col = f"{collection_name}_legacy"

                # Benchmark collection creation
                creation_results = {}
                for use_hybrid in [True, False]:
                    col_name = hybrid_col if use_hybrid else legacy_col
                    creation_results[use_hybrid] = self.benchmark_collection_creation(
                        col_name, use_hybrid
                    )

                # Benchmark insertion (only for hybrid collections for efficiency)
                insertion_results = self.benchmark_insertion_performance(
                    hybrid_col, documents
                )

                # Benchmark various search types
                search_results = self.benchmark_search_performance(
                    hybrid_col,
                    query_vector,
                    ["hybrid_dense", "hybrid_sparse", "hybrid_batch"],
                )

                # Legacy search comparison
                if legacy_col not in ["benchmark_1000_legacy"]:  # Limit to smaller sets
                    legacy_search = self.benchmark_search_performance(
                        legacy_col, query_vector, ["semantic_only"]
                    )
                    search_results.update(legacy_search)

                # Collect results
                report["results"][f"{count}_docs"] = {
                    "collection_creation": creation_results,
                    "insertion": insertion_results,
                    "search": search_results,
                    "memory_usage": {
                        "per_doc_bytes": len(json.dumps(documents[0]))
                        if documents
                        else 0
                    },
                }

                # Cleanup
                try:
                    self.wrapper.client.delete_collection(hybrid_col)
                    self.wrapper.client.delete_collection(legacy_col)
                except:
                    pass

                print(f"✅ Completed {count:,} documents test")

            except Exception as e:
                print(f"❌ Failed for {count:,} documents: {e}")
                report["results"][f"{count}_docs"] = {"error": str(e)}

        return report

    def generate_performance_report(self, benchmark_results: Dict[str, Any]) -> str:
        """Generate human-readable performance report."""

        report = []
        report.append("=" * 80)
        report.append("Hybrid Search Performance Report")
        report.append("=" * 80)
        report.append("")

        for count, results in benchmark_results["results"].items():
            if "error" in results:
                continue

            doc_count = int(count.split("_")[0])

            report.append(f"📊 Results for {doc_count:,} documents:")
            report.append("-" * 40)

            # Insertion performance
            insertion = results["insertion"]
            if insertion.get("success"):
                report.append(
                    f"  📥 Insertion: {insertion['docs_per_second']:.1f} docs/sec"
                )

            # Collection creation
            creation = results["collection_creation"]
            for hybrid_str, creation_result in creation.items():
                type_name = "Hybrid" if hybrid_str else "Legacy"
                creation_time = creation_result.get("creation_time", 0)
                if creation_time > 0:
                    report.append(f"  🏗️  {type_name} Creation: {creation_time:.3f}s")

            # Search performance
            search = results["search"]
            for test_type, perf_data in search.items():
                if "avg_time" in perf_data:
                    report.append(f"  🔍 {test_type}: {perf_data['avg_time']:.4f}s avg")
                    report.append(
                        f"      ({perf_data['min_time']:.4f}s - {perf_data['max_time']:.4f}s)"
                    )

            report.append("")

        # Summary
        report.append("🎯 Performance Summary:")
        report.append("=" * 40)

        # Find best performing scenarios
        all_search_results = []
        for count, results in benchmark_results["results"].items():
            if "error" not in results and "search" in results:
                for test_type, data in results["search"].items():
                    if "avg_time" in data:
                        all_search_results.append((test_type, data["avg_time"]))

        if all_search_results:
            fastest = min(all_search_results, key=lambda x: x[1])
            report.append(f"  🏆 Fastest: {fastest[0]} at {fastest[1]:.4f}s")

        return "\n".join(report)


def run_benchmark_cli():
    """Command line interface for running benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Qdrant Hybrid Search benchmarks")
    parser.add_argument("--docs", nargs="+", type=int, default=[100, 500, 1000, 5000])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", help="Output JSON file")

    args = parser.parse_args()

    benchmark = HybridSearchBenchmark(device=args.device)

    print("🧪 Running hybrid search performance benchmarks...")

    try:
        results = benchmark.run_comprehensive_benchmark(args.docs)
        report = benchmark.generate_performance_report(results)

        print("\n" + report)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n📋 Results saved to {args.output}")

    except KeyboardInterrupt:
        print("\n⏹️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")


if __name__ == "__main__":
    run_benchmark_cli()
