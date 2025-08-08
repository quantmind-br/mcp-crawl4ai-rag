# Ruby test fixture for Tree-sitter parsing tests.
#
# This file contains various Ruby language constructs to test comprehensive parsing.

require 'json'
require 'time'
require 'logger'
require 'concurrent'
require 'forwardable'

# Configuration class with validation and serialization
class ProcessorConfiguration
  attr_accessor :name, :debug, :timeout_ms, :options

  def initialize(name: 'default', debug: false, timeout_ms: 5000)
    self.name = name
    self.debug = debug
    self.timeout_ms = timeout_ms
    @options = {}
  end

  def name=(value)
    raise ArgumentError, 'Configuration name cannot be empty' if value.nil? || value.strip.empty?
    @name = value
  end

  def timeout_ms=(value)
    raise ArgumentError, 'Timeout must be greater than zero' if value <= 0
    @timeout_ms = value
  end

  def set_option(key, value)
    @options[key] = value
    self
  end

  def get_option(key, default = nil)
    @options[key] || default
  end

  def valid?
    !@name.nil? && !@name.strip.empty? && @timeout_ms > 0
  end

  def to_h
    {
      name: @name,
      debug: @debug,
      timeout_ms: @timeout_ms,
      options: @options
    }
  end

  def to_json(*args)
    to_h.to_json(*args)
  end

  def self.from_hash(data)
    config = new(
      name: data[:name] || data['name'] || 'default',
      debug: data[:debug] || data['debug'] || false,
      timeout_ms: data[:timeout_ms] || data['timeout_ms'] || 5000
    )

    options = data[:options] || data['options'] || {}
    options.each { |key, value| config.set_option(key, value) }

    config
  end

  def self.create_default(name)
    new(name: name)
  end

  def self.create_debug(name)
    new(name: name, debug: true)
  end
end

# Custom exception classes
class ProcessingError < StandardError
  INVALID_INPUT = 1
  PROCESSING_FAILED = 2
  TIMEOUT = 3
  RESOURCE_UNAVAILABLE = 4

  attr_reader :item, :error_code

  def initialize(message, error_code: PROCESSING_FAILED, item: nil)
    super(message)
    @error_code = error_code
    @item = item
  end
end

# Statistics container for processing operations
class ProcessingStats
  attr_reader :total_processed, :error_count, :config_name, :total_processing_time

  def initialize(total_processed: 0, error_count: 0, config_name: '', total_processing_time: 0.0)
    @total_processed = total_processed
    @error_count = error_count
    @config_name = config_name
    @total_processing_time = total_processing_time
  end

  def to_h
    {
      total_processed: @total_processed,
      error_count: @error_count,
      config_name: @config_name,
      total_processing_time: @total_processing_time
    }
  end

  def to_json(*args)
    to_h.to_json(*args)
  end
end

# Mixin module for logging functionality
module Loggable
  def log_debug(message)
    return unless @config&.debug

    puts "[DEBUG] #{@config.name}: #{message}"
  end

  def log_info(message)
    puts "[INFO] #{message}"
  end

  def log_error(message)
    warn "[ERROR] #{message}"
  end
end

# Base processor module defining the interface
module Processor
  def process_item(item)
    raise NotImplementedError, 'Subclasses must implement process_item'
  end

  def process(items)
    raise NotImplementedError, 'Subclasses must implement process'
  end

  def get_stats
    raise NotImplementedError, 'Subclasses must implement get_stats'
  end

  def reset
    raise NotImplementedError, 'Subclasses must implement reset'
  end
end

# Abstract base class for processors
class BaseProcessor
  include Processor
  include Loggable

  def initialize(config)
    raise ProcessingError.new('Invalid configuration', error_code: ProcessingError::INVALID_INPUT) unless config.valid?

    @config = config
    @results = []
    @error_count = 0
    @total_processing_time = 0.0
    @mutex = Mutex.new
  end

  protected

  def validate_item(item)
    item.is_a?(String) && !item.strip.empty? && item.length <= 1000
  end

  def add_result(result)
    @mutex.synchronize do
      @results << result
    end
  end

  def increment_error_count
    @mutex.synchronize do
      @error_count += 1
    end
  end

  def handle_error(error, item)
    increment_error_count
    log_error("Error processing '#{item}': #{error.message}") if @config.debug
  end

  public

  def process(items)
    start_time = Time.now
    processed = []

    items.each do |item|
      begin
        raise ProcessingError.new("Invalid item: #{item}", error_code: ProcessingError::INVALID_INPUT, item: item) unless validate_item(item)

        result = process_item(item)
        processed << result
        add_result(result)

        log_debug("Processed: #{item} -> #{result}")
      rescue ProcessingError => e
        handle_error(e, item)
        raise e
      rescue StandardError => e
        processing_error = ProcessingError.new("Processing failed for item: #{item}", error_code: ProcessingError::PROCESSING_FAILED, item: item)
        handle_error(processing_error, item)
        raise processing_error
      end
    end

    @total_processing_time += Time.now - start_time
    processed
  end

  def get_stats
    @mutex.synchronize do
      ProcessingStats.new(
        total_processed: @results.length,
        error_count: @error_count,
        config_name: @config.name,
        total_processing_time: @total_processing_time
      )
    end
  end

  def reset
    @mutex.synchronize do
      @results.clear
      @error_count = 0
      @total_processing_time = 0.0
    end
  end
end

# Concrete string processor implementation
class StringProcessor < BaseProcessor
  def process_item(item)
    # Simulate processing time
    sleep(0.001) # 1ms

    result = "processed_#{item.strip.downcase}"
    log_debug("Processed: #{item} -> #{result}")
    result
  end

  # Process items with custom transformation block
  def process_with_transform(items, &transform)
    results = []

    items.each do |item|
      next unless validate_item(item)

      begin
        processed = process_item(item)
        results << transform.call(processed)
      rescue StandardError => e
        increment_error_count
        log_error("Transform error for '#{item}': #{e.message}") if @config.debug
      end
    end

    results
  end

  # Process items with filtering
  def process_with_filter(items, &filter)
    filtered_items = items.select { |item| validate_item(item) && filter.call(item) }
    process(filtered_items)
  end

  # Process items asynchronously using threads
  def process_async(items, thread_count: 4)
    items_queue = Queue.new
    results_queue = Queue.new
    
    items.each { |item| items_queue << item }
    thread_count.times { items_queue << :stop }

    threads = Array.new(thread_count) do
      Thread.new do
        loop do
          item = items_queue.pop
          break if item == :stop

          begin
            result = process_item(item) if validate_item(item)
            results_queue << { success: true, result: result, item: item }
            add_result(result)
          rescue StandardError => e
            increment_error_count
            results_queue << { success: false, error: e, item: item }
          end
        end
      end
    end

    threads.each(&:join)

    results = []
    items.length.times do
      result_data = results_queue.pop
      results << result_data[:result] if result_data[:success]
    end

    results
  end

  # Process items with Enumerable methods
  def process_enumerable(items)
    items
      .lazy
      .select { |item| validate_item(item) }
      .map { |item| process_item(item) }
      .to_a
  end
end

# Batch processor using delegation
class BatchProcessor
  extend Forwardable

  def_delegators :@processor, :get_stats, :reset

  def initialize(processor, batch_size: 10)
    @processor = processor
    @batch_size = [batch_size, 1].max
    @batch_results = []
  end

  def process_batches(items)
    all_results = []
    items.each_slice(@batch_size).with_index do |batch, index|
      begin
        batch_results = @processor.process(batch)
        @batch_results[index] = batch_results
        all_results.concat(batch_results)
      rescue StandardError => e
        warn "Batch #{index} failed: #{e.message}"
        @batch_results[index] = []
      end
    end

    all_results
  end

  def batch_count
    @batch_results.length
  end

  def batch_results
    @batch_results.dup
  end
end

# Singleton factory class
class ProcessorFactory
  @instance = nil

  def self.instance
    @instance ||= new
  end

  private_class_method :new

  def create_string_processor(name, debug: false)
    config = ProcessorConfiguration.new(name: name, debug: debug)
    StringProcessor.new(config)
  end

  def create_batch_processor(processor, batch_size: 10)
    BatchProcessor.new(processor, batch_size: batch_size)
  end

  def create_from_config(config_data)
    config = ProcessorConfiguration.from_hash(config_data)
    StringProcessor.new(config)
  end

  # Convenience class methods
  def self.create_string_processor(name, debug: false)
    instance.create_string_processor(name, debug: debug)
  end

  def self.create_batch_processor(processor, batch_size: 10)
    instance.create_batch_processor(processor, batch_size: batch_size)
  end
end

# Utility module with helper methods
module ProcessorUtils
  module_function

  def generate_test_data(count, prefix: 'item')
    (0...count).map { |i| format('%s_%03d', prefix, i) }
  end

  def validate_items(items)
    items.is_a?(Array) && !items.empty? && items.all? { |item| item.is_a?(String) && !item.strip.empty? }
  end

  def print_results(results, title: 'Results')
    puts "#{title} (#{results.length} items):"
    results.each { |result| puts "  #{result}" }
  end

  def measure_execution_time
    start_time = Time.now
    result = yield
    execution_time = Time.now - start_time
    
    {
      result: result,
      execution_time: execution_time
    }
  end

  def load_config_from_file(file_path)
    raise ArgumentError, "Configuration file not found: #{file_path}" unless File.exist?(file_path)

    json_content = File.read(file_path)
    data = JSON.parse(json_content, symbolize_names: true)
    ProcessorConfiguration.from_hash(data)
  rescue JSON::ParserError => e
    raise StandardError, "Invalid JSON in configuration file: #{e.message}"
  end
end

# Enhanced processor with additional features
class EnhancedStringProcessor < StringProcessor
  def initialize(config, logger: nil)
    super(config)
    @logger = logger || Logger.new($stdout)
    @logger.level = config.debug ? Logger::DEBUG : Logger::INFO
  end

  def process_with_logging(items)
    @logger.info("Starting processing with #{items.length} items")
    
    begin
      results = process(items)
      @logger.info("Processing completed successfully with #{results.length} results")
      results
    rescue StandardError => e
      @logger.error("Processing failed: #{e.message}")
      raise e
    end
  end

  protected

  def handle_error(error, item)
    super
    @logger.error("Processing failed for '#{item}': #{error.message}")
  end
end

# Metaprogramming examples
class DynamicProcessor
  def self.define_processor_method(method_name, &block)
    define_method(method_name, &block)
  end

  def method_missing(method_name, *args, &block)
    if method_name.to_s.start_with?('process_')
      operation = method_name.to_s.sub('process_', '')
      puts "Dynamic processing with operation: #{operation}"
      args.first.map { |item| "#{operation}_#{item}" }
    else
      super
    end
  end

  def respond_to_missing?(method_name, include_private = false)
    method_name.to_s.start_with?('process_') || super
  end
end

# Define a processor method dynamically
DynamicProcessor.define_processor_method(:process_upcase) do |items|
  items.map(&:upcase)
end

# Constants and class variables
class ProcessorConstants
  VERSION = '1.0.0'.freeze
  DEFAULT_TIMEOUT = 5000
  @@instance_count = 0

  def initialize
    @@instance_count += 1
  end

  def self.instance_count
    @@instance_count
  end
end

# Main execution method
def main
  puts 'Starting Ruby processor application...'

  begin
    # Create configuration
    config = ProcessorConfiguration.new(name: 'main', debug: true, timeout_ms: 10_000)
    config.set_option('key1', 'value1')

    # Create processor
    logger = Logger.new($stdout)
    processor = EnhancedStringProcessor.new(config, logger: logger)

    # Generate test data
    test_data = ProcessorUtils.generate_test_data(5, prefix: 'item')

    unless ProcessorUtils.validate_items(test_data)
      raise ProcessingError.new('Invalid test data', error_code: ProcessingError::INVALID_INPUT)
    end

    # Synchronous processing with timing
    puts 'Processing synchronously...'
    sync_result = ProcessorUtils.measure_execution_time do
      processor.process_with_logging(test_data)
    end
    ProcessorUtils.print_results(sync_result[:result], title: 'Synchronous Results')
    puts format('Synchronous processing took: %.3f seconds', sync_result[:execution_time])

    # Processing with transformation using block
    puts 'Processing with transformation...'
    transform_results = processor.process_with_transform(test_data) { |item| item.upcase }
    ProcessorUtils.print_results(transform_results, title: 'Transform Results')

    # Processing with filtering
    puts 'Processing with filtering...'
    filter_results = processor.process_with_filter(test_data) { |item| item.include?('0') || item.include?('2') }
    ProcessorUtils.print_results(filter_results, title: 'Filter Results')

    # Asynchronous processing
    puts 'Processing asynchronously...'
    async_results = processor.process_async(test_data, thread_count: 2)
    ProcessorUtils.print_results(async_results, title: 'Async Results')

    # Enumerable processing
    puts 'Processing with enumerable methods...'
    enum_results = processor.process_enumerable(test_data)
    ProcessorUtils.print_results(enum_results, title: 'Enumerable Results')

    # Batch processing
    puts 'Processing in batches...'
    batch_processor = ProcessorFactory.create_batch_processor(processor, batch_size: 2)
    batch_results = batch_processor.process_batches(test_data)
    ProcessorUtils.print_results(batch_results, title: 'Batch Results')
    puts "Number of batches processed: #{batch_processor.batch_count}"

    # Dynamic processor example
    puts 'Dynamic processor example...'
    dynamic_processor = DynamicProcessor.new
    dynamic_results = dynamic_processor.process_upcase(['hello', 'world'])
    ProcessorUtils.print_results(dynamic_results, title: 'Dynamic Results')
    
    # Method missing example
    missing_results = dynamic_processor.process_custom(['test', 'method'])
    ProcessorUtils.print_results(missing_results, title: 'Method Missing Results')

    # Get statistics
    stats = processor.get_stats
    puts "Statistics: #{stats.to_json}"

    # Instance count example
    puts "Processor constant instances created: #{ProcessorConstants.instance_count}"

    puts 'Ruby processor application completed successfully.'

  rescue ProcessingError => e
    puts "Processing error: #{e.message} (Code: #{e.error_code}, Item: #{e.item})"
    exit(1)
  rescue StandardError => e
    puts "Unexpected error: #{e.message}"
    puts e.backtrace if $DEBUG
    exit(1)
  end
end

# Execute main method if script is run directly
main if __FILE__ == $PROGRAM_NAME