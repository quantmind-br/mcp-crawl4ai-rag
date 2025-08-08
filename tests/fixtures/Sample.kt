/**
 * Kotlin test fixture for Tree-sitter parsing tests.
 * 
 * This file contains various Kotlin language constructs to test comprehensive parsing.
 */

package com.example.processor

import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.time.Instant
import java.time.Duration
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.random.Random

/**
 * Data class for processor configuration with validation
 */
@Serializable
data class ProcessorConfiguration(
    val name: String = "default",
    val debug: Boolean = false,
    val timeoutMs: Long = 5000L,
    val options: Map<String, String> = emptyMap()
) {
    init {
        require(name.isNotBlank()) { "Configuration name cannot be empty" }
        require(timeoutMs > 0) { "Timeout must be greater than zero" }
    }
    
    fun validate(): Boolean = name.isNotBlank() && timeoutMs > 0
    
    companion object {
        fun createDefault(name: String) = ProcessorConfiguration(name = name)
        
        fun createDebug(name: String) = ProcessorConfiguration(
            name = name,
            debug = true
        )
        
        fun fromJson(json: String): ProcessorConfiguration = 
            Json.decodeFromString(json)
    }
    
    fun toJson(): String = Json.encodeToString(this)
}

/**
 * Sealed class hierarchy for processing errors
 */
sealed class ProcessingError(
    message: String,
    val item: String? = null,
    cause: Throwable? = null
) : Exception(message, cause) {
    
    class InvalidInput(item: String, message: String = "Invalid input") : 
        ProcessingError(message, item)
    
    class ProcessingFailed(item: String, message: String, cause: Throwable? = null) : 
        ProcessingError("Processing failed for item '$item': $message", item, cause)
    
    class Timeout(message: String = "Operation timed out") : 
        ProcessingError(message)
    
    class ResourceUnavailable(message: String = "Resource unavailable") : 
        ProcessingError(message)
    
    object Unknown : ProcessingError("Unknown error occurred")
}

/**
 * Data class for processing statistics
 */
@Serializable
data class ProcessingStats(
    val totalProcessed: Int = 0,
    val errorCount: Int = 0,
    val configName: String = "",
    val totalProcessingTimeMs: Long = 0L,
    val lastProcessed: String? = null
) {
    fun toJson(): String = Json.encodeToString(this)
    
    companion object {
        fun fromJson(json: String): ProcessingStats = 
            Json.decodeFromString(json)
    }
}

/**
 * Generic interface for data processors
 */
interface Processor<TInput, TOutput> {
    suspend fun processItem(item: TInput): TOutput
    suspend fun process(items: List<TInput>): List<TOutput>
    fun processSync(items: List<TInput>): List<TOutput>
    fun getStats(): ProcessingStats
    fun reset()
}

/**
 * Extension functions for collections
 */
fun <T> List<T>.chunked(size: Int, transform: (List<T>) -> List<T>): List<T> =
    this.chunked(size).flatMap(transform)

fun <T> List<T>.processInParallel(processor: suspend (T) -> T): List<T> = runBlocking {
    this@processInParallel.map { async { processor(it) } }.awaitAll()
}

/**
 * Abstract base class for processors with common functionality
 */
abstract class BaseProcessor<TInput, TOutput>(
    protected val config: ProcessorConfiguration
) : Processor<TInput, TOutput> {
    
    protected val results = mutableListOf<TOutput>()
    protected val errorCount = AtomicInteger(0)
    protected val totalProcessingTime = AtomicLong(0L)
    protected val lastProcessed = AtomicReference<String?>(null)
    
    init {
        require(config.validate()) { "Invalid configuration provided" }
    }
    
    protected abstract suspend fun processItemImpl(item: TInput): TOutput
    protected abstract fun validateItem(item: TInput): Boolean
    
    override suspend fun processItem(item: TInput): TOutput {
        if (!validateItem(item)) {
            throw ProcessingError.InvalidInput(item.toString())
        }
        
        return try {
            val startTime = System.currentTimeMillis()
            val result = processItemImpl(item)
            val processingTime = System.currentTimeMillis() - startTime
            
            totalProcessingTime.addAndGet(processingTime)
            
            synchronized(results) {
                results.add(result)
            }
            
            lastProcessed.set(item.toString())
            
            if (config.debug) {
                logDebug("Processed: $item -> $result")
            }
            
            result
        } catch (e: Exception) {
            errorCount.incrementAndGet()
            throw ProcessingError.ProcessingFailed(item.toString(), e.message ?: "Unknown error", e)
        }
    }
    
    override suspend fun process(items: List<TInput>): List<TOutput> {
        val processed = mutableListOf<TOutput>()
        
        items.forEach { item ->
            try {
                val result = processItem(item)
                processed.add(result)
            } catch (e: ProcessingError) {
                handleError(e, item)
                throw e
            }
        }
        
        return processed
    }
    
    override fun processSync(items: List<TInput>): List<TOutput> = runBlocking {
        process(items)
    }
    
    // Coroutine-based parallel processing
    suspend fun processParallel(
        items: List<TInput>,
        concurrency: Int = 4
    ): List<TOutput> = withContext(Dispatchers.Default) {
        items.chunked(concurrency).flatMap { chunk ->
            chunk.map { item ->
                async { processItem(item) }
            }.awaitAll()
        }
    }
    
    // Flow-based streaming processing
    fun processAsFlow(items: List<TInput>): Flow<TOutput> = flow {
        items.forEach { item ->
            emit(processItem(item))
        }
    }.flowOn(Dispatchers.Default)
    
    override fun getStats(): ProcessingStats {
        synchronized(results) {
            return ProcessingStats(
                totalProcessed = results.size,
                errorCount = errorCount.get(),
                configName = config.name,
                totalProcessingTimeMs = totalProcessingTime.get(),
                lastProcessed = lastProcessed.get()
            )
        }
    }
    
    override fun reset() {
        synchronized(results) {
            results.clear()
        }
        errorCount.set(0)
        totalProcessingTime.set(0L)
        lastProcessed.set(null)
    }
    
    protected open fun handleError(error: ProcessingError, item: TInput) {
        if (config.debug) {
            logError("Error processing '$item': ${error.message}")
        }
    }
    
    protected fun logDebug(message: String) {
        if (config.debug) {
            println("[DEBUG] ${config.name}: $message")
        }
    }
    
    protected fun logError(message: String) {
        System.err.println("[ERROR] ${config.name}: $message")
    }
}

/**
 * Concrete string processor implementation
 */
class StringProcessor(config: ProcessorConfiguration) : BaseProcessor<String, String>(config) {
    
    override suspend fun processItemImpl(item: String): String {
        // Simulate async processing time
        delay(1L) // 1ms
        
        val result = "processed_${item.trim().lowercase()}"
        logDebug("Processed: $item -> $result")
        return result
    }
    
    override fun validateItem(item: String): Boolean =
        item.isNotBlank() && item.length <= 1000
    
    // Higher-order function with lambda
    suspend fun processWithTransform(
        items: List<String>,
        transform: suspend (String) -> String
    ): List<String> {
        return items.mapNotNull { item ->
            try {
                if (validateItem(item)) {
                    val processed = processItemImpl(item)
                    transform(processed)
                } else null
            } catch (e: Exception) {
                errorCount.incrementAndGet()
                null
            }
        }
    }
    
    // Extension function usage
    fun processWithFilter(items: List<String>, predicate: (String) -> Boolean): List<String> =
        items.filter { validateItem(it) && predicate(it) }
            .map { runBlocking { processItemImpl(it) } }
    
    // Inline function with reified type parameter
    inline fun <reified T> processTyped(items: List<Any>): List<String> =
        items.filterIsInstance<T>()
            .map { it.toString() }
            .filter(::validateItem)
            .map { runBlocking { processItemImpl(it) } }
}

/**
 * Generic batch processor using delegation
 */
class BatchProcessor<TInput, TOutput>(
    private val processor: Processor<TInput, TOutput>,
    private val batchSize: Int = 10
) : Processor<TInput, TOutput> by processor {
    
    private val batchResults = ConcurrentHashMap<Int, List<TOutput>>()
    
    suspend fun processBatches(items: List<TInput>): List<TOutput> {
        val allResults = mutableListOf<TOutput>()
        
        items.chunked(batchSize).forEachIndexed { index, batch ->
            try {
                val batchResults = processor.process(batch)
                this.batchResults[index] = batchResults
                allResults.addAll(batchResults)
            } catch (e: Exception) {
                System.err.println("Batch $index failed: ${e.message}")
                this.batchResults[index] = emptyList()
            }
        }
        
        return allResults
    }
    
    fun getBatchCount(): Int = batchResults.size
    fun getBatchResults(): Map<Int, List<TOutput>> = batchResults.toMap()
}

/**
 * Object (singleton) factory for creating processors
 */
object ProcessorFactory {
    
    fun createStringProcessor(name: String, debug: Boolean = false): StringProcessor {
        val config = ProcessorConfiguration(name = name, debug = debug)
        return StringProcessor(config)
    }
    
    fun <TInput, TOutput> createBatchProcessor(
        processor: Processor<TInput, TOutput>,
        batchSize: Int = 10
    ): BatchProcessor<TInput, TOutput> = BatchProcessor(processor, batchSize)
    
    fun createFromConfig(configJson: String): StringProcessor {
        val config = ProcessorConfiguration.fromJson(configJson)
        return StringProcessor(config)
    }
    
    // Generic factory method with type parameter
    inline fun <reified T : BaseProcessor<String, String>> create(config: ProcessorConfiguration): T {
        return when (T::class) {
            StringProcessor::class -> StringProcessor(config) as T
            else -> throw IllegalArgumentException("Unsupported processor type: ${T::class}")
        }
    }
}

/**
 * Utility object with extension functions and helper methods
 */
object ProcessorUtils {
    
    fun generateTestData(count: Int, prefix: String = "item"): List<String> =
        (0 until count).map { i -> "${prefix}_${i.toString().padStart(3, '0')}" }
    
    fun validateItems(items: List<String>): Boolean =
        items.isNotEmpty() && items.all { it.isNotBlank() }
    
    fun printResults(results: List<Any>, title: String = "Results") {
        println("$title (${results.size} items):")
        results.forEach { result ->
            println("  $result")
        }
    }
    
    suspend fun <T> measureExecutionTime(block: suspend () -> T): Pair<T, Duration> {
        val startTime = Instant.now()
        val result = block()
        val executionTime = Duration.between(startTime, Instant.now())
        return result to executionTime
    }
    
    // Extension property
    val List<String>.wordCount: Int
        get() = this.sumOf { it.split(" ").size }
    
    // Extension function with receiver
    fun List<String>.processInBatches(
        batchSize: Int,
        processor: StringProcessor
    ): List<String> = this.chunked(batchSize) { batch ->
        processor.processSync(batch)
    }
}

/**
 * Enum class for processing modes
 */
enum class ProcessingMode(val displayName: String, val parallelism: Int) {
    SEQUENTIAL("Sequential", 1),
    PARALLEL("Parallel", 4),
    BATCH("Batch", 10);
    
    companion object {
        fun fromString(mode: String): ProcessingMode? =
            values().find { it.name.equals(mode, ignoreCase = true) }
    }
}

/**
 * Interface with default methods
 */
interface Configurable {
    val configuration: ProcessorConfiguration
    
    fun isDebugEnabled(): Boolean = configuration.debug
    
    fun getTimeout(): Long = configuration.timeoutMs
    
    fun logConfiguration() {
        println("Configuration: ${configuration.toJson()}")
    }
}

/**
 * Class implementing multiple interfaces
 */
class EnhancedStringProcessor(
    override val configuration: ProcessorConfiguration
) : BaseProcessor<String, String>(configuration), Configurable {
    
    override suspend fun processItemImpl(item: String): String {
        delay(Random.nextLong(1, 5)) // Variable processing time
        return "enhanced_processed_${item.trim().lowercase()}"
    }
    
    override fun validateItem(item: String): Boolean =
        item.isNotBlank() && item.length in 1..1000
    
    suspend fun processWithMode(items: List<String>, mode: ProcessingMode): List<String> {
        logConfiguration()
        
        return when (mode) {
            ProcessingMode.SEQUENTIAL -> process(items)
            ProcessingMode.PARALLEL -> processParallel(items, mode.parallelism)
            ProcessingMode.BATCH -> {
                val batchProcessor = BatchProcessor(this, mode.parallelism)
                batchProcessor.processBatches(items)
            }
        }
    }
}

/**
 * Generic class with type constraints
 */
class TypedProcessor<T>(
    private val processor: StringProcessor,
    private val converter: (T) -> String,
    private val reverseConverter: (String) -> T
) where T : Any {
    
    suspend fun process(items: List<T>): List<T> {
        val stringItems = items.map(converter)
        val processedStrings = processor.process(stringItems)
        return processedStrings.map(reverseConverter)
    }
}

/**
 * Annotation class for marking experimental features
 */
@Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.RUNTIME)
annotation class ExperimentalProcessor(val version: String = "1.0")

/**
 * Experimental processor with annotation
 */
@ExperimentalProcessor(version = "2.0")
class ExperimentalAsyncProcessor(config: ProcessorConfiguration) : BaseProcessor<String, String>(config) {
    
    override suspend fun processItemImpl(item: String): String {
        // Experimental async processing with coroutines
        return withContext(Dispatchers.IO) {
            delay(Random.nextLong(1, 10))
            "experimental_${item.reversed()}_${Random.nextInt(1000)}"
        }
    }
    
    override fun validateItem(item: String): Boolean = item.isNotEmpty()
}

/**
 * Main function demonstrating Kotlin features
 */
suspend fun main() {
    println("Starting Kotlin processor application...")
    
    try {
        // Create configuration using data class
        val config = ProcessorConfiguration(
            name = "main",
            debug = true,
            timeoutMs = 10000L,
            options = mapOf("key1" to "value1", "key2" to "value2")
        )
        
        // Create enhanced processor
        val processor = EnhancedStringProcessor(config)
        
        // Generate test data using utility object
        val testData = ProcessorUtils.generateTestData(5, "item")
        
        if (!ProcessorUtils.validateItems(testData)) {
            throw ProcessingError.InvalidInput("", "Invalid test data")
        }
        
        // Sequential processing with time measurement
        println("Processing sequentially...")
        val (seqResults, seqTime) = ProcessorUtils.measureExecutionTime {
            processor.processWithMode(testData, ProcessingMode.SEQUENTIAL)
        }
        ProcessorUtils.printResults(seqResults, "Sequential Results")
        println("Sequential processing took: ${seqTime.toMillis()}ms")
        
        // Parallel processing
        println("Processing in parallel...")
        val (parallelResults, parallelTime) = ProcessorUtils.measureExecutionTime {
            processor.processWithMode(testData, ProcessingMode.PARALLEL)
        }
        ProcessorUtils.printResults(parallelResults, "Parallel Results")
        println("Parallel processing took: ${parallelTime.toMillis()}ms")
        
        // Batch processing
        println("Processing in batches...")
        val batchResults = processor.processWithMode(testData, ProcessingMode.BATCH)
        ProcessorUtils.printResults(batchResults, "Batch Results")
        
        // Processing with transformation using lambda
        println("Processing with transformation...")
        val transformResults = processor.processWithTransform(testData) { result ->
            result.uppercase()
        }
        ProcessorUtils.printResults(transformResults, "Transform Results")
        
        // Processing with filtering
        println("Processing with filtering...")
        val filterResults = processor.processWithFilter(testData) { item ->
            item.contains("0") || item.contains("2")
        }
        ProcessorUtils.printResults(filterResults, "Filter Results")
        
        // Flow-based processing
        println("Processing with Flow...")
        val flowResults = mutableListOf<String>()
        processor.processAsFlow(testData).collect { result ->
            flowResults.add(result)
        }
        ProcessorUtils.printResults(flowResults, "Flow Results")
        
        // Typed processing example
        println("Typed processing example...")
        val typedProcessor = TypedProcessor<Int>(
            processor = StringProcessor(config),
            converter = { it.toString() },
            reverseConverter = { it.filter { char -> char.isDigit() }.toIntOrNull() ?: 0 }
        )
        val typedResults = typedProcessor.process(listOf(123, 456, 789))
        ProcessorUtils.printResults(typedResults, "Typed Results")
        
        // Extension function usage
        println("Extension function usage...")
        val extensionResults = testData.processInBatches(2, StringProcessor(config))
        ProcessorUtils.printResults(extensionResults, "Extension Results")
        
        // Get statistics
        val stats = processor.getStats()
        println("Statistics: ${stats.toJson()}")
        
        // Experimental processor
        println("Experimental processing...")
        val experimentalProcessor = ExperimentalAsyncProcessor(config)
        val experimentalResults = experimentalProcessor.process(testData.take(3))
        ProcessorUtils.printResults(experimentalResults, "Experimental Results")
        
        println("Kotlin processor application completed successfully.")
        
    } catch (error: ProcessingError) {
        System.err.println("Processing error: ${error.message} (Item: ${error.item})")
        throw error
    } catch (error: Exception) {
        System.err.println("Unexpected error: ${error.message}")
        error.printStackTrace()
        throw error
    }
}