"""
Enhanced Neo4j bulk operations for high-performance repository indexing.

This module provides optimized bulk operations using UNWIND patterns for
inserting large amounts of data into Neo4j with minimal transaction overhead.
Designed to handle 5000+ operations per transaction for maximum performance.
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from neo4j import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class BulkOperationResult:
    """Result from a bulk operation."""

    operation_type: str
    records_processed: int
    records_created: int
    records_updated: int
    processing_time_seconds: float
    errors: List[str]


class Neo4jBulkProcessor:
    """
    High-performance Neo4j bulk processor using UNWIND operations.

    Optimized for large-scale repository indexing with batch sizes of 5000+
    operations per transaction for maximum throughput.
    """

    def __init__(self, batch_size: int = 5000):
        """
        Initialize bulk processor.

        Args:
            batch_size: Number of operations per batch (default: 5000)
        """
        self.batch_size = batch_size
        self.stats = {
            "total_operations": 0,
            "total_batches": 0,
            "total_time": 0.0,
            "operation_results": [],
        }

    async def bulk_create_files(
        self,
        session: AsyncSession,
        file_data: List[Dict[str, Any]],
        repo_name: str,
    ) -> BulkOperationResult:
        """
        Bulk create file nodes using UNWIND for optimal performance.

        Args:
            session: Neo4j async session
            file_data: List of file data dictionaries
            repo_name: Repository name for relationship creation

        Returns:
            BulkOperationResult with operation statistics
        """
        import time

        start_time = time.time()

        try:
            # Process in batches to avoid memory issues
            total_created = 0
            total_updated = 0
            errors = []

            for i in range(0, len(file_data), self.batch_size):
                batch = file_data[i : i + self.batch_size]

                # Bulk create file nodes with UNWIND
                query = """
                UNWIND $batch AS file_data
                MERGE (f:File {path: file_data.path})
                ON CREATE SET 
                    f.name = file_data.name,
                    f.language = file_data.language,
                    f.file_type = file_data.file_type,
                    f.line_count = file_data.line_count,
                    f.created_at = datetime()
                ON MATCH SET
                    f.language = file_data.language,
                    f.file_type = file_data.file_type,
                    f.line_count = file_data.line_count,
                    f.updated_at = datetime()
                RETURN f.path AS path, 
                       CASE WHEN f.created_at = datetime() THEN 'created' ELSE 'updated' END AS action
                """

                result = await session.run(query, batch=batch)
                records = await result.data()

                # Count created vs updated
                batch_created = sum(1 for r in records if r.get("action") == "created")
                batch_updated = len(records) - batch_created

                total_created += batch_created
                total_updated += batch_updated

                logger.debug(
                    f"File batch {i // self.batch_size + 1}: "
                    f"{batch_created} created, {batch_updated} updated"
                )

            # Create repository relationships in bulk
            if file_data:
                await self._bulk_create_repository_relationships(
                    session, file_data, repo_name
                )

            processing_time = time.time() - start_time

            result = BulkOperationResult(
                operation_type="bulk_create_files",
                records_processed=len(file_data),
                records_created=total_created,
                records_updated=total_updated,
                processing_time_seconds=processing_time,
                errors=errors,
            )

            self.stats["operation_results"].append(result)
            logger.info(
                f"Bulk file creation completed: {total_created} created, "
                f"{total_updated} updated in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Error in bulk file creation: {e}")
            raise

    async def bulk_create_classes(
        self,
        session: AsyncSession,
        class_data: List[Dict[str, Any]],
    ) -> BulkOperationResult:
        """
        Bulk create class nodes using UNWIND for optimal performance.

        Args:
            session: Neo4j async session
            class_data: List of class data dictionaries

        Returns:
            BulkOperationResult with operation statistics
        """
        import time

        start_time = time.time()

        try:
            total_created = 0
            total_updated = 0
            errors = []

            for i in range(0, len(class_data), self.batch_size):
                batch = class_data[i : i + self.batch_size]

                # Bulk create class nodes with UNWIND
                query = """
                UNWIND $batch AS class_data
                MERGE (c:Class {full_name: class_data.full_name})
                ON CREATE SET 
                    c.name = class_data.name,
                    c.line_start = class_data.line_start,
                    c.line_end = class_data.line_end,
                    c.docstring = class_data.docstring,
                    c.created_at = datetime()
                ON MATCH SET
                    c.line_start = class_data.line_start,
                    c.line_end = class_data.line_end,
                    c.docstring = class_data.docstring,
                    c.updated_at = datetime()
                RETURN c.full_name AS full_name,
                       CASE WHEN c.created_at = datetime() THEN 'created' ELSE 'updated' END AS action
                """

                result = await session.run(query, batch=batch)
                records = await result.data()

                batch_created = sum(1 for r in records if r.get("action") == "created")
                batch_updated = len(records) - batch_created

                total_created += batch_created
                total_updated += batch_updated

            # Create file-class relationships in bulk
            if class_data:
                await self._bulk_create_file_class_relationships(session, class_data)

            processing_time = time.time() - start_time

            result = BulkOperationResult(
                operation_type="bulk_create_classes",
                records_processed=len(class_data),
                records_created=total_created,
                records_updated=total_updated,
                processing_time_seconds=processing_time,
                errors=errors,
            )

            self.stats["operation_results"].append(result)
            logger.info(
                f"Bulk class creation completed: {total_created} created, "
                f"{total_updated} updated in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Error in bulk class creation: {e}")
            raise

    async def bulk_create_methods(
        self,
        session: AsyncSession,
        method_data: List[Dict[str, Any]],
    ) -> BulkOperationResult:
        """
        Bulk create method nodes using UNWIND for optimal performance.

        Args:
            session: Neo4j async session
            method_data: List of method data dictionaries

        Returns:
            BulkOperationResult with operation statistics
        """
        import time

        start_time = time.time()

        try:
            total_created = 0
            total_updated = 0
            errors = []

            for i in range(0, len(method_data), self.batch_size):
                batch = method_data[i : i + self.batch_size]

                # Bulk create method nodes with UNWIND
                query = """
                UNWIND $batch AS method_data
                MERGE (m:Method {method_id: method_data.method_id})
                ON CREATE SET 
                    m.name = method_data.name,
                    m.params_list = method_data.params_list,
                    m.params_detailed = method_data.params_detailed,
                    m.return_type = method_data.return_type,
                    m.args = method_data.args,
                    m.line_start = method_data.line_start,
                    m.line_end = method_data.line_end,
                    m.docstring = method_data.docstring,
                    m.class_name = method_data.class_name,
                    m.created_at = datetime()
                ON MATCH SET
                    m.params_list = method_data.params_list,
                    m.params_detailed = method_data.params_detailed,
                    m.return_type = method_data.return_type,
                    m.args = method_data.args,
                    m.line_start = method_data.line_start,
                    m.line_end = method_data.line_end,
                    m.docstring = method_data.docstring,
                    m.updated_at = datetime()
                RETURN m.method_id AS method_id,
                       CASE WHEN m.created_at = datetime() THEN 'created' ELSE 'updated' END AS action
                """

                result = await session.run(query, batch=batch)
                records = await result.data()

                batch_created = sum(1 for r in records if r.get("action") == "created")
                batch_updated = len(records) - batch_created

                total_created += batch_created
                total_updated += batch_updated

            # Create class-method relationships in bulk
            if method_data:
                await self._bulk_create_class_method_relationships(session, method_data)

            processing_time = time.time() - start_time

            result = BulkOperationResult(
                operation_type="bulk_create_methods",
                records_processed=len(method_data),
                records_created=total_created,
                records_updated=total_updated,
                processing_time_seconds=processing_time,
                errors=errors,
            )

            self.stats["operation_results"].append(result)
            logger.info(
                f"Bulk method creation completed: {total_created} created, "
                f"{total_updated} updated in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Error in bulk method creation: {e}")
            raise

    async def bulk_create_functions(
        self,
        session: AsyncSession,
        function_data: List[Dict[str, Any]],
    ) -> BulkOperationResult:
        """
        Bulk create function nodes using UNWIND for optimal performance.

        Args:
            session: Neo4j async session
            function_data: List of function data dictionaries

        Returns:
            BulkOperationResult with operation statistics
        """
        import time

        start_time = time.time()

        try:
            total_created = 0
            total_updated = 0
            errors = []

            for i in range(0, len(function_data), self.batch_size):
                batch = function_data[i : i + self.batch_size]

                # Bulk create function nodes with UNWIND
                query = """
                UNWIND $batch AS func_data
                MERGE (f:Function {func_id: func_data.func_id})
                ON CREATE SET 
                    f.name = func_data.name,
                    f.full_name = func_data.full_name,
                    f.params_list = func_data.params_list,
                    f.params_detailed = func_data.params_detailed,
                    f.return_type = func_data.return_type,
                    f.args = func_data.args,
                    f.line_start = func_data.line_start,
                    f.line_end = func_data.line_end,
                    f.docstring = func_data.docstring,
                    f.created_at = datetime()
                ON MATCH SET
                    f.params_list = func_data.params_list,
                    f.params_detailed = func_data.params_detailed,
                    f.return_type = func_data.return_type,
                    f.args = func_data.args,
                    f.line_start = func_data.line_start,
                    f.line_end = func_data.line_end,
                    f.docstring = func_data.docstring,
                    f.updated_at = datetime()
                RETURN f.func_id AS func_id,
                       CASE WHEN f.created_at = datetime() THEN 'created' ELSE 'updated' END AS action
                """

                result = await session.run(query, batch=batch)
                records = await result.data()

                batch_created = sum(1 for r in records if r.get("action") == "created")
                batch_updated = len(records) - batch_created

                total_created += batch_created
                total_updated += batch_updated

            # Create file-function relationships in bulk
            if function_data:
                await self._bulk_create_file_function_relationships(
                    session, function_data
                )

            processing_time = time.time() - start_time

            result = BulkOperationResult(
                operation_type="bulk_create_functions",
                records_processed=len(function_data),
                records_created=total_created,
                records_updated=total_updated,
                processing_time_seconds=processing_time,
                errors=errors,
            )

            self.stats["operation_results"].append(result)
            logger.info(
                f"Bulk function creation completed: {total_created} created, "
                f"{total_updated} updated in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Error in bulk function creation: {e}")
            raise

    async def bulk_create_imports(
        self,
        session: AsyncSession,
        import_data: List[Dict[str, Any]],
    ) -> BulkOperationResult:
        """
        Bulk create import relationships using UNWIND for optimal performance.

        Args:
            session: Neo4j async session
            import_data: List of import relationship data

        Returns:
            BulkOperationResult with operation statistics
        """
        import time

        start_time = time.time()

        try:
            total_created = 0
            errors = []

            for i in range(0, len(import_data), self.batch_size):
                batch = import_data[i : i + self.batch_size]

                # Bulk create import relationships with UNWIND
                query = """
                UNWIND $batch AS import_data
                MATCH (source:File {path: import_data.source_file})
                MATCH (target:File {path: import_data.target_file})
                MERGE (source)-[r:IMPORTS]->(target)
                ON CREATE SET r.import_name = import_data.import_name,
                             r.created_at = datetime()
                RETURN COUNT(r) AS relationships_created
                """

                result = await session.run(query, batch=batch)
                record = await result.single()

                if record:
                    total_created += record.get("relationships_created", 0)

            processing_time = time.time() - start_time

            result = BulkOperationResult(
                operation_type="bulk_create_imports",
                records_processed=len(import_data),
                records_created=total_created,
                records_updated=0,
                processing_time_seconds=processing_time,
                errors=errors,
            )

            self.stats["operation_results"].append(result)
            logger.info(
                f"Bulk import creation completed: {total_created} created "
                f"in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Error in bulk import creation: {e}")
            raise

    # Helper methods for relationship creation

    async def _bulk_create_repository_relationships(
        self,
        session: AsyncSession,
        file_data: List[Dict[str, Any]],
        repo_name: str,
    ):
        """Create repository-file relationships in bulk."""
        # First ensure repository exists
        await session.run(
            "MERGE (r:Repository {name: $repo_name}) ON CREATE SET r.created_at = datetime()",
            repo_name=repo_name,
        )

        # Create relationships in batches
        for i in range(0, len(file_data), self.batch_size):
            batch = file_data[i : i + self.batch_size]
            paths = [item["path"] for item in batch]

            query = """
            UNWIND $paths AS file_path
            MATCH (r:Repository {name: $repo_name})
            MATCH (f:File {path: file_path})
            MERGE (r)-[:CONTAINS]->(f)
            """

            await session.run(query, paths=paths, repo_name=repo_name)

    async def _bulk_create_file_class_relationships(
        self,
        session: AsyncSession,
        class_data: List[Dict[str, Any]],
    ):
        """Create file-class relationships in bulk."""
        for i in range(0, len(class_data), self.batch_size):
            batch = class_data[i : i + self.batch_size]

            query = """
            UNWIND $batch AS class_data
            MATCH (f:File {path: class_data.file_path})
            MATCH (c:Class {full_name: class_data.full_name})
            MERGE (f)-[:DEFINES]->(c)
            """

            await session.run(query, batch=batch)

    async def _bulk_create_class_method_relationships(
        self,
        session: AsyncSession,
        method_data: List[Dict[str, Any]],
    ):
        """Create class-method relationships in bulk."""
        for i in range(0, len(method_data), self.batch_size):
            batch = method_data[i : i + self.batch_size]

            query = """
            UNWIND $batch AS method_data
            MATCH (c:Class {full_name: method_data.class_full_name})
            MATCH (m:Method {method_id: method_data.method_id})
            MERGE (c)-[:HAS_METHOD]->(m)
            """

            await session.run(query, batch=batch)

    async def _bulk_create_file_function_relationships(
        self,
        session: AsyncSession,
        function_data: List[Dict[str, Any]],
    ):
        """Create file-function relationships in bulk."""
        for i in range(0, len(function_data), self.batch_size):
            batch = function_data[i : i + self.batch_size]

            query = """
            UNWIND $batch AS func_data
            MATCH (f:File {path: func_data.file_path})
            MATCH (func:Function {func_id: func_data.func_id})
            MERGE (f)-[:DEFINES]->(func)
            """

            await session.run(query, batch=batch)

    def get_bulk_stats(self) -> Dict[str, Any]:
        """Get bulk operation statistics."""
        return {
            "total_operations": self.stats["total_operations"],
            "total_batches": self.stats["total_batches"],
            "total_time": self.stats["total_time"],
            "average_batch_time": (
                self.stats["total_time"] / self.stats["total_batches"]
                if self.stats["total_batches"] > 0
                else 0
            ),
            "operation_results": self.stats["operation_results"],
            "batch_size": self.batch_size,
        }
