/**
 * Java test fixture for Tree-sitter parsing tests.
 * 
 * This file contains various Java language constructs to test comprehensive parsing.
 */

package com.example.fixtures;

import java.util.*;
import java.util.concurrent.*;
import java.util.function.*;
import java.util.stream.*;
import java.io.*;
import java.nio.file.*;
import javax.annotation.processing.*;
import org.springframework.stereotype.*;
import org.springframework.beans.factory.annotation.*;

/**
 * Configuration class with builder pattern.
 */
@Component
@ConfigurationProperties(prefix = "processor")
public class ProcessorConfiguration {
    private String name = "default";
    private boolean debug = false;
    private int timeout = 5000;
    private List<String> allowedTypes = new ArrayList<>();
    
    // Constructor
    public ProcessorConfiguration() {}
    
    public ProcessorConfiguration(String name, boolean debug) {
        this.name = name;
        this.debug = debug;
    }
    
    // Getters and setters
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    
    public boolean isDebug() { return debug; }
    public void setDebug(boolean debug) { this.debug = debug; }
    
    public int getTimeout() { return timeout; }
    public void setTimeout(int timeout) { this.timeout = timeout; }
    
    public List<String> getAllowedTypes() { return new ArrayList<>(allowedTypes); }
    public void setAllowedTypes(List<String> allowedTypes) { 
        this.allowedTypes = new ArrayList<>(allowedTypes); 
    }
    
    /**
     * Builder pattern implementation.
     */
    public static class Builder {
        private String name = "default";
        private boolean debug = false;
        private int timeout = 5000;
        private List<String> allowedTypes = new ArrayList<>();
        
        public Builder name(String name) {
            this.name = name;
            return this;
        }
        
        public Builder debug(boolean debug) {
            this.debug = debug;
            return this;
        }
        
        public Builder timeout(int timeout) {
            this.timeout = timeout;
            return this;
        }
        
        public Builder allowedType(String type) {
            this.allowedTypes.add(type);
            return this;
        }
        
        public ProcessorConfiguration build() {
            ProcessorConfiguration config = new ProcessorConfiguration(name, debug);
            config.setTimeout(timeout);
            config.setAllowedTypes(allowedTypes);
            return config;
        }
    }
    
    public static Builder builder() {
        return new Builder();
    }
}

/**
 * Abstract base class for data processors.
 */
public abstract class BaseDataProcessor<T> {
    protected final ProcessorConfiguration config;
    protected final List<T> results;
    protected int errorCount;
    
    public BaseDataProcessor(ProcessorConfiguration config) {
        this.config = Objects.requireNonNull(config, "Configuration cannot be null");
        this.results = Collections.synchronizedList(new ArrayList<>());
        this.errorCount = 0;
    }
    
    /**
     * Abstract method for processing individual items.
     * @param item The item to process
     * @return The processed result
     * @throws ProcessingException if processing fails
     */
    public abstract T processItem(String item) throws ProcessingException;
    
    /**
     * Process a collection of items.
     * @param items Items to process
     * @return List of processed results
     */
    public List<T> process(Collection<String> items) {
        Objects.requireNonNull(items, "Items cannot be null");
        
        List<T> processed = new ArrayList<>();
        
        for (String item : items) {
            try {
                T result = processItem(item);
                processed.add(result);
                results.add(result);
                
                if (config.isDebug()) {
                    System.out.println("Processed: " + item + " -> " + result);
                }
            } catch (ProcessingException e) {
                errorCount++;
                handleError(e, item);
            }
        }
        
        return processed;
    }
    
    /**
     * Process items asynchronously using CompletableFuture.
     * @param items Items to process
     * @return CompletableFuture with list of results
     */
    public CompletableFuture<List<T>> processAsync(Collection<String> items) {
        return CompletableFuture.supplyAsync(() -> {
            return items.parallelStream()
                    .map(item -> {
                        try {
                            return processItem(item);
                        } catch (ProcessingException e) {
                            handleError(e, item);
                            return null;
                        }
                    })
                    .filter(Objects::nonNull)
                    .collect(Collectors.toList());
        });
    }
    
    /**
     * Handle processing errors.
     * @param error The error that occurred
     * @param item The item that caused the error
     */
    protected void handleError(ProcessingException error, String item) {
        if (config.isDebug()) {
            System.err.println("Error processing '" + item + "': " + error.getMessage());
        }
    }
    
    /**
     * Get processing statistics.
     * @return Map of statistics
     */
    public Map<String, Object> getStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("totalProcessed", results.size());
        stats.put("errorCount", errorCount);
        stats.put("configName", config.getName());
        return stats;
    }
    
    // Protected methods for subclasses
    protected boolean validateItem(String item) {
        return item != null && !item.trim().isEmpty();
    }
    
    protected void logDebug(String message) {
        if (config.isDebug()) {
            System.out.println("[DEBUG] " + message);
        }
    }
}

/**
 * Concrete implementation of BaseDataProcessor for string processing.
 */
@Service
public class StringDataProcessor extends BaseDataProcessor<String> {
    
    private final ExecutorService executorService;
    
    @Autowired
    public StringDataProcessor(ProcessorConfiguration config) {
        super(config);
        this.executorService = Executors.newFixedThreadPool(4);
    }
    
    // Alternative constructor for testing
    public StringDataProcessor(ProcessorConfiguration config, ExecutorService executorService) {
        super(config);
        this.executorService = executorService;
    }
    
    @Override
    public String processItem(String item) throws ProcessingException {
        if (!validateItem(item)) {
            throw new ProcessingException("Invalid item: " + item);
        }
        
        String processed = "processed_" + item.trim().toLowerCase();
        logDebug("Processed item: " + item + " -> " + processed);
        return processed;
    }
    
    /**
     * Process items with custom transformation function.
     * @param items Items to process
     * @param transformer Custom transformation function
     * @return List of transformed results
     */
    public <R> List<R> processWithTransform(Collection<String> items, 
                                          Function<String, R> transformer) {
        return items.stream()
                .filter(this::validateItem)
                .map(item -> {
                    try {
                        String processed = processItem(item);
                        return transformer.apply(processed);
                    } catch (ProcessingException e) {
                        handleError(e, item);
                        return null;
                    }
                })
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
    }
    
    /**
     * Batch process items with size limits.
     * @param items Items to process
     * @param batchSize Maximum batch size
     * @return List of batch results
     */
    public List<CompletableFuture<List<String>>> processBatches(List<String> items, 
                                                               int batchSize) {
        List<CompletableFuture<List<String>>> futures = new ArrayList<>();
        
        for (int i = 0; i < items.size(); i += batchSize) {
            int endIndex = Math.min(i + batchSize, items.size());
            List<String> batch = items.subList(i, endIndex);
            
            CompletableFuture<List<String>> future = CompletableFuture
                    .supplyAsync(() -> process(batch), executorService);
            futures.add(future);
        }
        
        return futures;
    }
    
    /**
     * Shutdown the processor and cleanup resources.
     */
    public void shutdown() {
        executorService.shutdown();
        try {
            if (!executorService.awaitTermination(5, TimeUnit.SECONDS)) {
                executorService.shutdownNow();
            }
        } catch (InterruptedException e) {
            executorService.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}

/**
 * Custom exception for processing errors.
 */
public class ProcessingException extends Exception {
    private final String item;
    private final ErrorCode errorCode;
    
    public ProcessingException(String message) {
        super(message);
        this.item = null;
        this.errorCode = ErrorCode.UNKNOWN;
    }
    
    public ProcessingException(String message, String item) {
        super(message);
        this.item = item;
        this.errorCode = ErrorCode.PROCESSING_FAILED;
    }
    
    public ProcessingException(String message, String item, ErrorCode errorCode) {
        super(message);
        this.item = item;
        this.errorCode = errorCode;
    }
    
    public ProcessingException(String message, Throwable cause) {
        super(message, cause);
        this.item = null;
        this.errorCode = ErrorCode.UNKNOWN;
    }
    
    public String getItem() { return item; }
    public ErrorCode getErrorCode() { return errorCode; }
    
    public enum ErrorCode {
        UNKNOWN,
        INVALID_INPUT,
        PROCESSING_FAILED,
        TIMEOUT,
        RESOURCE_UNAVAILABLE
    }
}

/**
 * Utility class with static methods.
 */
public final class ProcessorUtils {
    
    // Private constructor to prevent instantiation
    private ProcessorUtils() {
        throw new UnsupportedOperationException("Utility class");
    }
    
    /**
     * Create processor with default configuration.
     * @return New StringDataProcessor instance
     */
    public static StringDataProcessor createDefaultProcessor() {
        ProcessorConfiguration config = ProcessorConfiguration.builder()
                .name("default")
                .debug(false)
                .timeout(5000)
                .build();
        return new StringDataProcessor(config);
    }
    
    /**
     * Validate collection of items.
     * @param items Items to validate
     * @return true if all items are valid
     */
    public static boolean validateItems(Collection<String> items) {
        return items != null && 
               !items.isEmpty() && 
               items.stream().allMatch(item -> item != null && !item.trim().isEmpty());
    }
    
    /**
     * Generate test data.
     * @param count Number of items to generate
     * @param prefix Prefix for generated items
     * @return List of test items
     */
    public static List<String> generateTestData(int count, String prefix) {
        return IntStream.range(0, count)
                .mapToObj(i -> String.format("%s_%03d", prefix, i))
                .collect(Collectors.toList());
    }
    
    /**
     * Combine multiple processor results.
     * @param results Multiple result lists
     * @return Combined result list
     */
    @SafeVarargs
    public static <T> List<T> combineResults(List<T>... results) {
        return Arrays.stream(results)
                .filter(Objects::nonNull)
                .flatMap(List::stream)
                .collect(Collectors.toList());
    }
}

/**
 * Main class with application entry point.
 */
public class ProcessorApplication {
    
    /**
     * Main method for testing the processor.
     * @param args Command line arguments
     */
    public static void main(String[] args) {
        System.out.println("Starting Processor Application...");
        
        try {
            // Create configuration
            ProcessorConfiguration config = ProcessorConfiguration.builder()
                    .name("main")
                    .debug(true)
                    .timeout(10000)
                    .allowedType("string")
                    .build();
            
            // Create processor
            StringDataProcessor processor = new StringDataProcessor(config);
            
            // Generate test data
            List<String> testData = ProcessorUtils.generateTestData(5, "item");
            
            if (!ProcessorUtils.validateItems(testData)) {
                System.err.println("Invalid test data");
                return;
            }
            
            // Process synchronously
            System.out.println("Processing synchronously...");
            List<String> results = processor.process(testData);
            System.out.println("Results: " + results);
            
            // Process asynchronously
            System.out.println("Processing asynchronously...");
            CompletableFuture<List<String>> futureResults = processor.processAsync(testData);
            List<String> asyncResults = futureResults.get(5, TimeUnit.SECONDS);
            System.out.println("Async results: " + asyncResults);
            
            // Get statistics
            Map<String, Object> stats = processor.getStats();
            System.out.println("Statistics: " + stats);
            
            // Process with transformation
            List<Integer> lengths = processor.processWithTransform(testData, String::length);
            System.out.println("Lengths: " + lengths);
            
            // Cleanup
            processor.shutdown();
            
        } catch (Exception e) {
            System.err.println("Application error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
        
        System.out.println("Application completed successfully.");
    }
}