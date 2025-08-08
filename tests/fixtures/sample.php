<?php
/**
 * PHP test fixture for Tree-sitter parsing tests.
 * 
 * This file contains various PHP language constructs to test comprehensive parsing.
 */

declare(strict_types=1);

namespace ProcessorFramework;

require_once __DIR__ . '/vendor/autoload.php';

use Exception;
use InvalidArgumentException;
use RuntimeException;
use JsonSerializable;
use Countable;
use IteratorAggregate;
use ArrayIterator;
use Closure;

/**
 * Configuration class with validation and serialization
 */
class ProcessorConfiguration implements JsonSerializable
{
    private string $name;
    private bool $debug;
    private int $timeoutMs;
    private array $options;

    public function __construct(string $name = 'default', bool $debug = false, int $timeoutMs = 5000)
    {
        $this->setName($name);
        $this->debug = $debug;
        $this->setTimeoutMs($timeoutMs);
        $this->options = [];
    }

    public function getName(): string
    {
        return $this->name;
    }

    public function setName(string $name): self
    {
        if (empty(trim($name))) {
            throw new InvalidArgumentException('Configuration name cannot be empty');
        }
        $this->name = $name;
        return $this;
    }

    public function isDebug(): bool
    {
        return $this->debug;
    }

    public function setDebug(bool $debug): self
    {
        $this->debug = $debug;
        return $this;
    }

    public function getTimeoutMs(): int
    {
        return $this->timeoutMs;
    }

    public function setTimeoutMs(int $timeoutMs): self
    {
        if ($timeoutMs <= 0) {
            throw new InvalidArgumentException('Timeout must be greater than zero');
        }
        $this->timeoutMs = $timeoutMs;
        return $this;
    }

    public function getOptions(): array
    {
        return $this->options;
    }

    public function setOption(string $key, string $value): self
    {
        $this->options[$key] = $value;
        return $this;
    }

    public function getOption(string $key, ?string $default = null): ?string
    {
        return $this->options[$key] ?? $default;
    }

    public function validate(): bool
    {
        return !empty($this->name) && $this->timeoutMs > 0;
    }

    public function jsonSerialize(): array
    {
        return [
            'name' => $this->name,
            'debug' => $this->debug,
            'timeoutMs' => $this->timeoutMs,
            'options' => $this->options,
        ];
    }

    public static function fromArray(array $data): self
    {
        $config = new self(
            $data['name'] ?? 'default',
            $data['debug'] ?? false,
            $data['timeoutMs'] ?? 5000
        );

        if (isset($data['options']) && is_array($data['options'])) {
            foreach ($data['options'] as $key => $value) {
                $config->setOption((string) $key, (string) $value);
            }
        }

        return $config;
    }
}

/**
 * Custom exception for processing errors
 */
class ProcessingException extends Exception
{
    public const INVALID_INPUT = 1;
    public const PROCESSING_FAILED = 2;
    public const TIMEOUT = 3;
    public const RESOURCE_UNAVAILABLE = 4;

    private ?string $item;

    public function __construct(string $message, int $code = self::PROCESSING_FAILED, ?string $item = null, ?Exception $previous = null)
    {
        parent::__construct($message, $code, $previous);
        $this->item = $item;
    }

    public function getItem(): ?string
    {
        return $this->item;
    }
}

/**
 * Statistics container for processing operations
 */
class ProcessingStats implements JsonSerializable
{
    public function __construct(
        private int $totalProcessed = 0,
        private int $errorCount = 0,
        private string $configName = '',
        private float $totalProcessingTime = 0.0
    ) {}

    public function getTotalProcessed(): int
    {
        return $this->totalProcessed;
    }

    public function getErrorCount(): int
    {
        return $this->errorCount;
    }

    public function getConfigName(): string
    {
        return $this->configName;
    }

    public function getTotalProcessingTime(): float
    {
        return $this->totalProcessingTime;
    }

    public function jsonSerialize(): array
    {
        return [
            'totalProcessed' => $this->totalProcessed,
            'errorCount' => $this->errorCount,
            'configName' => $this->configName,
            'totalProcessingTime' => $this->totalProcessingTime,
        ];
    }
}

/**
 * Interface for data processors
 */
interface ProcessorInterface
{
    public function processItem(string $item): string;
    public function process(array $items): array;
    public function getStats(): ProcessingStats;
    public function reset(): void;
}

/**
 * Abstract base class for processors
 */
abstract class BaseProcessor implements ProcessorInterface
{
    protected ProcessorConfiguration $config;
    protected array $results = [];
    protected int $errorCount = 0;
    protected float $totalProcessingTime = 0.0;

    public function __construct(ProcessorConfiguration $config)
    {
        if (!$config->validate()) {
            throw new ProcessingException('Invalid configuration', ProcessingException::INVALID_INPUT);
        }
        $this->config = $config;
    }

    abstract public function processItem(string $item): string;

    protected function validateItem(string $item): bool
    {
        return !empty(trim($item)) && strlen($item) <= 1000;
    }

    public function process(array $items): array
    {
        $startTime = microtime(true);
        $processed = [];

        foreach ($items as $item) {
            try {
                if (!$this->validateItem($item)) {
                    throw new ProcessingException("Invalid item: {$item}", ProcessingException::INVALID_INPUT, $item);
                }

                $result = $this->processItem($item);
                $processed[] = $result;
                $this->results[] = $result;

                if ($this->config->isDebug()) {
                    $this->logDebug("Processed: {$item} -> {$result}");
                }
            } catch (ProcessingException $e) {
                $this->errorCount++;
                $this->handleError($e, $item);
                // Re-throw to allow caller to handle
                throw $e;
            } catch (Exception $e) {
                $this->errorCount++;
                $processingException = new ProcessingException(
                    "Processing failed for item: {$item}",
                    ProcessingException::PROCESSING_FAILED,
                    $item,
                    $e
                );
                $this->handleError($processingException, $item);
                throw $processingException;
            }
        }

        $this->totalProcessingTime += microtime(true) - $startTime;
        return $processed;
    }

    protected function handleError(ProcessingException $error, string $item): void
    {
        if ($this->config->isDebug()) {
            error_log("Error processing '{$item}': {$error->getMessage()}");
        }
    }

    protected function logDebug(string $message): void
    {
        if ($this->config->isDebug()) {
            echo "[DEBUG] {$this->config->getName()}: {$message}\n";
        }
    }

    public function getStats(): ProcessingStats
    {
        return new ProcessingStats(
            count($this->results),
            $this->errorCount,
            $this->config->getName(),
            $this->totalProcessingTime
        );
    }

    public function reset(): void
    {
        $this->results = [];
        $this->errorCount = 0;
        $this->totalProcessingTime = 0.0;
    }
}

/**
 * Concrete string processor implementation
 */
class StringProcessor extends BaseProcessor
{
    public function processItem(string $item): string
    {
        // Simulate processing time
        usleep(1000); // 1ms

        $processed = 'processed_' . strtolower(trim($item));
        $this->logDebug("Processed: {$item} -> {$processed}");

        return $processed;
    }

    /**
     * Process items with custom transformation
     */
    public function processWithTransform(array $items, callable $transform): array
    {
        $results = [];

        foreach ($items as $item) {
            if ($this->validateItem($item)) {
                try {
                    $processed = $this->processItem($item);
                    $results[] = $transform($processed);
                } catch (Exception $e) {
                    $this->errorCount++;
                    if ($this->config->isDebug()) {
                        error_log("Transform error for '{$item}': {$e->getMessage()}");
                    }
                }
            }
        }

        return $results;
    }

    /**
     * Process items with filtering
     */
    public function processWithFilter(array $items, callable $filter): array
    {
        $filteredItems = array_filter($items, fn($item) => $this->validateItem($item) && $filter($item));
        return $this->process($filteredItems);
    }

    /**
     * Process items asynchronously using generator
     */
    public function processGenerator(array $items): \Generator
    {
        foreach ($items as $item) {
            try {
                if ($this->validateItem($item)) {
                    yield $this->processItem($item);
                }
            } catch (Exception $e) {
                $this->errorCount++;
                yield null;
            }
        }
    }
}

/**
 * Batch processor for handling large datasets
 */
class BatchProcessor implements Countable, IteratorAggregate
{
    private ProcessorInterface $processor;
    private int $batchSize;
    private array $batchResults = [];

    public function __construct(ProcessorInterface $processor, int $batchSize = 10)
    {
        $this->processor = $processor;
        $this->batchSize = max(1, $batchSize);
    }

    public function processBatches(array $items): array
    {
        $allResults = [];
        $batches = array_chunk($items, $this->batchSize);

        foreach ($batches as $batchIndex => $batch) {
            try {
                $batchResults = $this->processor->process($batch);
                $this->batchResults[$batchIndex] = $batchResults;
                $allResults = array_merge($allResults, $batchResults);
            } catch (Exception $e) {
                error_log("Batch {$batchIndex} failed: {$e->getMessage()}");
                $this->batchResults[$batchIndex] = [];
            }
        }

        return $allResults;
    }

    public function count(): int
    {
        return count($this->batchResults);
    }

    public function getIterator(): ArrayIterator
    {
        return new ArrayIterator($this->batchResults);
    }

    public function getBatchResults(): array
    {
        return $this->batchResults;
    }
}

/**
 * Factory class for creating processors
 */
class ProcessorFactory
{
    public static function createStringProcessor(string $name, bool $debug = false): StringProcessor
    {
        $config = new ProcessorConfiguration($name, $debug);
        return new StringProcessor($config);
    }

    public static function createBatchProcessor(ProcessorInterface $processor, int $batchSize = 10): BatchProcessor
    {
        return new BatchProcessor($processor, $batchSize);
    }

    public static function createFromConfig(array $configData): StringProcessor
    {
        $config = ProcessorConfiguration::fromArray($configData);
        return new StringProcessor($config);
    }
}

/**
 * Utility class with static helper methods
 */
class ProcessorUtils
{
    /**
     * Generate test data
     */
    public static function generateTestData(int $count, string $prefix = 'item'): array
    {
        return array_map(
            fn($i) => sprintf('%s_%03d', $prefix, $i),
            range(0, $count - 1)
        );
    }

    /**
     * Validate array of items
     */
    public static function validateItems(array $items): bool
    {
        return !empty($items) && array_reduce(
            $items,
            fn($carry, $item) => $carry && is_string($item) && !empty(trim($item)),
            true
        );
    }

    /**
     * Print results with formatting
     */
    public static function printResults(array $results, string $title = 'Results'): void
    {
        echo "{$title} (" . count($results) . " items):\n";
        foreach ($results as $result) {
            echo "  {$result}\n";
        }
    }

    /**
     * Measure execution time
     */
    public static function measureExecutionTime(callable $callback): array
    {
        $startTime = microtime(true);
        $result = $callback();
        $executionTime = microtime(true) - $startTime;

        return [
            'result' => $result,
            'executionTime' => $executionTime,
        ];
    }

    /**
     * Create configuration from JSON file
     */
    public static function loadConfigFromFile(string $filePath): ProcessorConfiguration
    {
        if (!file_exists($filePath)) {
            throw new InvalidArgumentException("Configuration file not found: {$filePath}");
        }

        $jsonContent = file_get_contents($filePath);
        if ($jsonContent === false) {
            throw new RuntimeException("Failed to read configuration file: {$filePath}");
        }

        $data = json_decode($jsonContent, true);
        if (json_last_error() !== JSON_ERROR_NONE) {
            throw new RuntimeException("Invalid JSON in configuration file: " . json_last_error_msg());
        }

        return ProcessorConfiguration::fromArray($data);
    }
}

/**
 * Trait for logging functionality
 */
trait LoggingTrait
{
    protected function log(string $level, string $message, array $context = []): void
    {
        $timestamp = date('Y-m-d H:i:s');
        $contextStr = !empty($context) ? ' ' . json_encode($context) : '';
        echo "[{$timestamp}] {$level}: {$message}{$contextStr}\n";
    }

    protected function logInfo(string $message, array $context = []): void
    {
        $this->log('INFO', $message, $context);
    }

    protected function logError(string $message, array $context = []): void
    {
        $this->log('ERROR', $message, $context);
    }

    protected function logDebug(string $message, array $context = []): void
    {
        $this->log('DEBUG', $message, $context);
    }
}

/**
 * Enhanced processor with logging capabilities
 */
class EnhancedStringProcessor extends StringProcessor
{
    use LoggingTrait;

    protected function handleError(ProcessingException $error, string $item): void
    {
        parent::handleError($error, $item);
        $this->logError("Processing failed", [
            'item' => $item,
            'error' => $error->getMessage(),
            'code' => $error->getCode(),
        ]);
    }

    public function processWithLogging(array $items): array
    {
        $this->logInfo("Starting processing", ['itemCount' => count($items)]);
        
        try {
            $results = $this->process($items);
            $this->logInfo("Processing completed successfully", ['resultCount' => count($results)]);
            return $results;
        } catch (Exception $e) {
            $this->logError("Processing failed", ['error' => $e->getMessage()]);
            throw $e;
        }
    }
}

/**
 * Anonymous function examples and closures
 */
function demonstrateClosure(): Closure
{
    $multiplier = 2;
    
    return function(string $item) use ($multiplier): string {
        return str_repeat($item, $multiplier);
    };
}

/**
 * Main execution function
 */
function main(): void
{
    echo "Starting PHP processor application...\n";

    try {
        // Create configuration
        $config = new ProcessorConfiguration('main', true, 10000);
        $config->setOption('key1', 'value1');

        // Create processor
        $processor = new EnhancedStringProcessor($config);

        // Generate test data
        $testData = ProcessorUtils::generateTestData(5, 'item');

        if (!ProcessorUtils::validateItems($testData)) {
            throw new ProcessingException('Invalid test data', ProcessingException::INVALID_INPUT);
        }

        // Synchronous processing with timing
        echo "Processing synchronously...\n";
        $syncResult = ProcessorUtils::measureExecutionTime(
            fn() => $processor->processWithLogging($testData)
        );
        ProcessorUtils::printResults($syncResult['result'], 'Synchronous Results');
        echo sprintf("Synchronous processing took: %.3f seconds\n", $syncResult['executionTime']);

        // Processing with transformation
        echo "Processing with transformation...\n";
        $transformResults = $processor->processWithTransform($testData, fn($item) => strtoupper($item));
        ProcessorUtils::printResults($transformResults, 'Transform Results');

        // Processing with filtering
        echo "Processing with filtering...\n";
        $filterResults = $processor->processWithFilter(
            $testData,
            fn($item) => str_contains($item, '0') || str_contains($item, '2')
        );
        ProcessorUtils::printResults($filterResults, 'Filter Results');

        // Generator processing
        echo "Processing with generator...\n";
        $generatorResults = [];
        foreach ($processor->processGenerator($testData) as $result) {
            if ($result !== null) {
                $generatorResults[] = $result;
            }
        }
        ProcessorUtils::printResults($generatorResults, 'Generator Results');

        // Batch processing
        echo "Processing in batches...\n";
        $batchProcessor = ProcessorFactory::createBatchProcessor($processor, 2);
        $batchResults = $batchProcessor->processBatches($testData);
        ProcessorUtils::printResults($batchResults, 'Batch Results');
        echo "Number of batches processed: " . count($batchProcessor) . "\n";

        // Closure example
        echo "Processing with closure...\n";
        $closure = demonstrateClosure();
        $closureResults = array_map($closure, ['test', 'closure']);
        ProcessorUtils::printResults($closureResults, 'Closure Results');

        // Get statistics
        $stats = $processor->getStats();
        echo "Statistics: " . json_encode($stats, JSON_PRETTY_PRINT) . "\n";

        echo "PHP processor application completed successfully.\n";

    } catch (ProcessingException $e) {
        echo "Processing error: {$e->getMessage()} (Code: {$e->getCode()}, Item: {$e->getItem()})\n";
        exit(1);
    } catch (Exception $e) {
        echo "Unexpected error: {$e->getMessage()}\n";
        exit(1);
    }
}

// Execute main function if script is run directly
if (basename(__FILE__) === basename($_SERVER['SCRIPT_NAME'])) {
    main();
}