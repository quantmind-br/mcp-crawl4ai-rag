/**
 * C# test fixture for Tree-sitter parsing tests.
 * 
 * This file contains various C# language constructs to test comprehensive parsing.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using System.ComponentModel.DataAnnotations;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;

namespace ProcessorFramework
{
    /// <summary>
    /// Configuration record for immutable data
    /// </summary>
    public record ProcessorConfiguration
    {
        [Required]
        public string Name { get; init; } = "default";
        
        public bool Debug { get; init; } = false;
        
        [Range(1, int.MaxValue)]
        public int TimeoutMs { get; init; } = 5000;
        
        public Dictionary<string, string> Options { get; init; } = new();
        
        /// <summary>
        /// Create a configuration with builder pattern
        /// </summary>
        public static ProcessorConfiguration Create(string name) =>
            new() { Name = name };
        
        /// <summary>
        /// Validate the configuration
        /// </summary>
        public bool IsValid() => 
            !string.IsNullOrWhiteSpace(Name) && TimeoutMs > 0;
    }

    /// <summary>
    /// Custom exceptions for processing errors
    /// </summary>
    public class ProcessingException : Exception
    {
        public string? Item { get; }
        public ErrorCode Code { get; }
        
        public ProcessingException(string message, ErrorCode code = ErrorCode.Unknown) 
            : base(message)
        {
            Code = code;
        }
        
        public ProcessingException(string message, string item, ErrorCode code = ErrorCode.ProcessingFailed) 
            : base(message)
        {
            Item = item;
            Code = code;
        }
        
        public ProcessingException(string message, Exception innerException) 
            : base(message, innerException)
        {
            Code = ErrorCode.Unknown;
        }
        
        public enum ErrorCode
        {
            Unknown = 0,
            InvalidInput = 1,
            ProcessingFailed = 2,
            Timeout = 3,
            ResourceUnavailable = 4
        }
    }

    /// <summary>
    /// Generic interface for data processors
    /// </summary>
    public interface IProcessor<TInput, TOutput>
    {
        TOutput ProcessItem(TInput item);
        Task<TOutput> ProcessItemAsync(TInput item);
        IEnumerable<TOutput> Process(IEnumerable<TInput> items);
        Task<IEnumerable<TOutput>> ProcessAsync(IEnumerable<TInput> items);
        ProcessingStats GetStats();
        void Reset();
    }

    /// <summary>
    /// Statistics for processing operations
    /// </summary>
    public record ProcessingStats
    {
        public int TotalProcessed { get; init; }
        public int ErrorCount { get; init; }
        public string ConfigName { get; init; } = string.Empty;
        public DateTime LastProcessed { get; init; }
        public TimeSpan TotalProcessingTime { get; init; }
    }

    /// <summary>
    /// Abstract base class for processors with common functionality
    /// </summary>
    public abstract class BaseProcessor<TInput, TOutput> : IProcessor<TInput, TOutput>
    {
        protected readonly ProcessorConfiguration _config;
        protected readonly ILogger<BaseProcessor<TInput, TOutput>> _logger;
        protected readonly ConcurrentBag<TOutput> _results = new();
        protected int _errorCount = 0;
        protected DateTime _lastProcessed = DateTime.MinValue;
        private readonly object _statsLock = new();

        protected BaseProcessor(ProcessorConfiguration config, ILogger<BaseProcessor<TInput, TOutput>>? logger = null)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger<BaseProcessor<TInput, TOutput>>.Instance;
            
            if (!config.IsValid())
            {
                throw new ProcessingException("Invalid configuration", ProcessingException.ErrorCode.InvalidInput);
            }
        }

        public abstract TOutput ProcessItem(TInput item);
        
        protected virtual bool ValidateItem(TInput item) => item is not null;
        
        public virtual async Task<TOutput> ProcessItemAsync(TInput item)
        {
            return await Task.Run(() => ProcessItem(item));
        }

        public virtual IEnumerable<TOutput> Process(IEnumerable<TInput> items)
        {
            ArgumentNullException.ThrowIfNull(items);
            
            var results = new List<TOutput>();
            var startTime = DateTime.UtcNow;
            
            foreach (var item in items)
            {
                try
                {
                    if (!ValidateItem(item))
                    {
                        throw new ProcessingException($"Invalid item: {item}", item?.ToString() ?? "null");
                    }
                    
                    var result = ProcessItem(item);
                    results.Add(result);
                    _results.Add(result);
                    
                    if (_config.Debug)
                    {
                        _logger.LogDebug("Processed item: {Item} -> {Result}", item, result);
                    }
                }
                catch (ProcessingException)
                {
                    Interlocked.Increment(ref _errorCount);
                    throw;
                }
                catch (Exception ex)
                {
                    Interlocked.Increment(ref _errorCount);
                    _logger.LogError(ex, "Error processing item: {Item}", item);
                    throw new ProcessingException($"Processing failed for item: {item}", ex);
                }
            }
            
            lock (_statsLock)
            {
                _lastProcessed = DateTime.UtcNow;
            }
            
            return results;
        }

        public virtual async Task<IEnumerable<TOutput>> ProcessAsync(IEnumerable<TInput> items)
        {
            ArgumentNullException.ThrowIfNull(items);
            
            var tasks = items.Select(ProcessItemAsync);
            var results = await Task.WhenAll(tasks);
            
            foreach (var result in results)
            {
                _results.Add(result);
            }
            
            lock (_statsLock)
            {
                _lastProcessed = DateTime.UtcNow;
            }
            
            return results;
        }

        public virtual ProcessingStats GetStats()
        {
            lock (_statsLock)
            {
                return new ProcessingStats
                {
                    TotalProcessed = _results.Count,
                    ErrorCount = _errorCount,
                    ConfigName = _config.Name,
                    LastProcessed = _lastProcessed
                };
            }
        }

        public virtual void Reset()
        {
            _results.Clear();
            Interlocked.Exchange(ref _errorCount, 0);
            
            lock (_statsLock)
            {
                _lastProcessed = DateTime.MinValue;
            }
        }

        protected void LogDebug(string message, params object[] args)
        {
            if (_config.Debug)
            {
                _logger.LogDebug(message, args);
            }
        }
    }

    /// <summary>
    /// Concrete string processor implementation
    /// </summary>
    public class StringProcessor : BaseProcessor<string, string>
    {
        public StringProcessor(ProcessorConfiguration config, ILogger<StringProcessor>? logger = null) 
            : base(config, logger) { }

        public override string ProcessItem(string item)
        {
            if (string.IsNullOrWhiteSpace(item))
            {
                throw new ProcessingException("Item cannot be null or whitespace", 
                                            item, ProcessingException.ErrorCode.InvalidInput);
            }
            
            // Simulate processing delay
            Task.Delay(1).Wait();
            
            var result = $"processed_{item.Trim().ToLowerInvariant()}";
            LogDebug("Processed: {Item} -> {Result}", item, result);
            
            return result;
        }

        protected override bool ValidateItem(string item)
        {
            return !string.IsNullOrEmpty(item) && item.Length <= 1000;
        }

        /// <summary>
        /// Process items with custom transformation
        /// </summary>
        public IEnumerable<TResult> ProcessWithTransform<TResult>(
            IEnumerable<string> items, 
            Func<string, TResult> transform)
        {
            ArgumentNullException.ThrowIfNull(items);
            ArgumentNullException.ThrowIfNull(transform);
            
            return from item in items
                   where ValidateItem(item)
                   let processed = ProcessItem(item)
                   select transform(processed);
        }

        /// <summary>
        /// Process items with filtering and transformation using LINQ
        /// </summary>
        public async Task<IEnumerable<TResult>> ProcessWithFilterAsync<TResult>(
            IEnumerable<string> items,
            Func<string, bool> filter,
            Func<string, TResult> transform)
        {
            ArgumentNullException.ThrowIfNull(items);
            ArgumentNullException.ThrowIfNull(filter);
            ArgumentNullException.ThrowIfNull(transform);
            
            var filteredItems = items.Where(filter);
            var processed = await ProcessAsync(filteredItems);
            
            return processed.Select(transform);
        }
    }

    /// <summary>
    /// Generic batch processor with configurable batch size
    /// </summary>
    public class BatchProcessor<TInput, TOutput> : IDisposable
    {
        private readonly IProcessor<TInput, TOutput> _processor;
        private readonly int _batchSize;
        private readonly SemaphoreSlim _semaphore;
        private bool _disposed = false;

        public BatchProcessor(IProcessor<TInput, TOutput> processor, int batchSize = 10)
        {
            _processor = processor ?? throw new ArgumentNullException(nameof(processor));
            _batchSize = Math.Max(1, batchSize);
            _semaphore = new SemaphoreSlim(Environment.ProcessorCount, Environment.ProcessorCount);
        }

        public async Task<IEnumerable<TOutput>> ProcessBatchesAsync(IEnumerable<TInput> items)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            
            var allResults = new ConcurrentBag<TOutput>();
            var batches = items.Chunk(_batchSize);
            var tasks = new List<Task>();
            
            foreach (var batch in batches)
            {
                var task = ProcessBatchAsync(batch, allResults);
                tasks.Add(task);
            }
            
            await Task.WhenAll(tasks);
            return allResults;
        }

        private async Task ProcessBatchAsync(IEnumerable<TInput> batch, ConcurrentBag<TOutput> results)
        {
            await _semaphore.WaitAsync();
            
            try
            {
                var batchResults = await _processor.ProcessAsync(batch);
                foreach (var result in batchResults)
                {
                    results.Add(result);
                }
            }
            finally
            {
                _semaphore.Release();
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed && disposing)
            {
                _semaphore?.Dispose();
                _disposed = true;
            }
        }
    }

    /// <summary>
    /// Factory for creating processors with dependency injection
    /// </summary>
    public static class ProcessorFactory
    {
        public static StringProcessor CreateStringProcessor(
            string name, 
            bool debug = false, 
            ILogger<StringProcessor>? logger = null)
        {
            var config = new ProcessorConfiguration 
            { 
                Name = name, 
                Debug = debug 
            };
            
            return new StringProcessor(config, logger);
        }

        public static BatchProcessor<string, string> CreateBatchStringProcessor(
            string name,
            int batchSize = 10,
            bool debug = false,
            ILogger<StringProcessor>? logger = null)
        {
            var processor = CreateStringProcessor(name, debug, logger);
            return new BatchProcessor<string, string>(processor, batchSize);
        }
    }

    /// <summary>
    /// Extension methods for IEnumerable
    /// </summary>
    public static class EnumerableExtensions
    {
        public static IEnumerable<IEnumerable<T>> Chunk<T>(this IEnumerable<T> source, int size)
        {
            using var enumerator = source.GetEnumerator();
            while (enumerator.MoveNext())
            {
                yield return GetChunk(enumerator, size);
            }
        }

        private static IEnumerable<T> GetChunk<T>(IEnumerator<T> enumerator, int size)
        {
            do
            {
                yield return enumerator.Current;
            } while (--size > 0 && enumerator.MoveNext());
        }
    }

    /// <summary>
    /// Utility class with helper methods
    /// </summary>
    public static class ProcessorUtils
    {
        /// <summary>
        /// Generate test data with specified count and prefix
        /// </summary>
        public static IEnumerable<string> GenerateTestData(int count, string prefix = "item")
        {
            return Enumerable.Range(0, count)
                           .Select(i => $"{prefix}_{i:D3}");
        }

        /// <summary>
        /// Validate collection of items
        /// </summary>
        public static bool ValidateItems(IEnumerable<string> items)
        {
            return items?.Any() == true && items.All(item => !string.IsNullOrWhiteSpace(item));
        }

        /// <summary>
        /// Print results to console with formatting
        /// </summary>
        public static void PrintResults<T>(IEnumerable<T> results, string title = "Results")
        {
            var resultsList = results.ToList();
            Console.WriteLine($"{title} ({resultsList.Count} items):");
            
            foreach (var result in resultsList)
            {
                Console.WriteLine($"  {result}");
            }
        }

        /// <summary>
        /// Measure execution time of an action
        /// </summary>
        public static async Task<TimeSpan> MeasureExecutionTimeAsync(Func<Task> action)
        {
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            await action();
            stopwatch.Stop();
            return stopwatch.Elapsed;
        }
    }

    /// <summary>
    /// Configuration builder with fluent interface
    /// </summary>
    public class ProcessorConfigurationBuilder
    {
        private string _name = "default";
        private bool _debug = false;
        private int _timeoutMs = 5000;
        private Dictionary<string, string> _options = new();

        public ProcessorConfigurationBuilder WithName(string name)
        {
            _name = name ?? throw new ArgumentNullException(nameof(name));
            return this;
        }

        public ProcessorConfigurationBuilder WithDebug(bool debug = true)
        {
            _debug = debug;
            return this;
        }

        public ProcessorConfigurationBuilder WithTimeout(int timeoutMs)
        {
            _timeoutMs = timeoutMs > 0 ? timeoutMs : throw new ArgumentOutOfRangeException(nameof(timeoutMs));
            return this;
        }

        public ProcessorConfigurationBuilder WithOption(string key, string value)
        {
            _options[key] = value;
            return this;
        }

        public ProcessorConfiguration Build()
        {
            return new ProcessorConfiguration
            {
                Name = _name,
                Debug = _debug,
                TimeoutMs = _timeoutMs,
                Options = new Dictionary<string, string>(_options)
            };
        }
    }

    /// <summary>
    /// Main program class demonstrating C# features
    /// </summary>
    public class Program
    {
        public static async Task Main(string[] args)
        {
            Console.WriteLine("Starting C# processor application...");
            
            try
            {
                // Create configuration using builder pattern
                var config = new ProcessorConfigurationBuilder()
                    .WithName("main")
                    .WithDebug(true)
                    .WithTimeout(10000)
                    .WithOption("key1", "value1")
                    .Build();

                // Create processor
                var processor = new StringProcessor(config);
                
                // Generate test data using LINQ
                var testData = ProcessorUtils.GenerateTestData(5, "item").ToList();
                
                if (!ProcessorUtils.ValidateItems(testData))
                {
                    throw new ProcessingException("Invalid test data");
                }
                
                // Synchronous processing
                Console.WriteLine("Processing synchronously...");
                var syncResults = processor.Process(testData);
                ProcessorUtils.PrintResults(syncResults, "Synchronous Results");
                
                // Asynchronous processing
                Console.WriteLine("Processing asynchronously...");
                var asyncExecutionTime = await ProcessorUtils.MeasureExecutionTimeAsync(async () =>
                {
                    var asyncResults = await processor.ProcessAsync(testData);
                    ProcessorUtils.PrintResults(asyncResults, "Asynchronous Results");
                });
                Console.WriteLine($"Async processing took: {asyncExecutionTime.TotalMilliseconds}ms");
                
                // Processing with transformation
                Console.WriteLine("Processing with transformation...");
                var transformResults = processor.ProcessWithTransform(testData, s => s.ToUpperInvariant());
                ProcessorUtils.PrintResults(transformResults, "Transform Results");
                
                // Processing with filtering
                Console.WriteLine("Processing with filtering...");
                var filterResults = await processor.ProcessWithFilterAsync(
                    testData,
                    item => item.Contains("0") || item.Contains("2"),
                    processed => $"filtered_{processed}");
                ProcessorUtils.PrintResults(filterResults, "Filter Results");
                
                // Batch processing
                Console.WriteLine("Processing in batches...");
                using var batchProcessor = ProcessorFactory.CreateBatchStringProcessor("batch", 2, true);
                var batchResults = await batchProcessor.ProcessBatchesAsync(testData);
                ProcessorUtils.PrintResults(batchResults, "Batch Results");
                
                // Get statistics
                var stats = processor.GetStats();
                Console.WriteLine($"Statistics: Processed={stats.TotalProcessed}, Errors={stats.ErrorCount}, Config={stats.ConfigName}");
                
                // JSON serialization example
                var statsJson = JsonConvert.SerializeObject(stats, Formatting.Indented);
                Console.WriteLine($"Stats JSON:\n{statsJson}");
                
                Console.WriteLine("C# processor application completed successfully.");
            }
            catch (ProcessingException ex)
            {
                Console.WriteLine($"Processing error: {ex.Message} (Code: {ex.Code}, Item: {ex.Item})");
                Environment.Exit(1);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Unexpected error: {ex.Message}");
                Environment.Exit(1);
            }
        }
    }
}