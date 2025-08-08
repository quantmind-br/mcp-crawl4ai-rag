/**
 * C++ test fixture for Tree-sitter parsing tests.
 * 
 * This file contains various C++ language constructs to test comprehensive parsing.
 */

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>
#include <thread>
#include <mutex>
#include <future>
#include <chrono>
#include <map>
#include <exception>

namespace ProcessorFramework {

// Forward declarations
template<typename T>
class DataProcessor;

class ProcessingException;

/**
 * Configuration class with builder pattern
 */
class ProcessorConfiguration {
private:
    std::string name_;
    bool debug_;
    int timeout_ms_;
    std::map<std::string, std::string> options_;

public:
    // Constructor with default values
    explicit ProcessorConfiguration(const std::string& name = "default")
        : name_(name), debug_(false), timeout_ms_(5000) {}
    
    // Copy constructor
    ProcessorConfiguration(const ProcessorConfiguration& other)
        : name_(other.name_), debug_(other.debug_), 
          timeout_ms_(other.timeout_ms_), options_(other.options_) {}
    
    // Move constructor
    ProcessorConfiguration(ProcessorConfiguration&& other) noexcept
        : name_(std::move(other.name_)), debug_(other.debug_),
          timeout_ms_(other.timeout_ms_), options_(std::move(other.options_)) {}
    
    // Assignment operators
    ProcessorConfiguration& operator=(const ProcessorConfiguration& other) {
        if (this != &other) {
            name_ = other.name_;
            debug_ = other.debug_;
            timeout_ms_ = other.timeout_ms_;
            options_ = other.options_;
        }
        return *this;
    }
    
    ProcessorConfiguration& operator=(ProcessorConfiguration&& other) noexcept {
        if (this != &other) {
            name_ = std::move(other.name_);
            debug_ = other.debug_;
            timeout_ms_ = other.timeout_ms_;
            options_ = std::move(other.options_);
        }
        return *this;
    }
    
    // Destructor
    virtual ~ProcessorConfiguration() = default;
    
    // Getters
    const std::string& getName() const { return name_; }
    bool isDebug() const { return debug_; }
    int getTimeoutMs() const { return timeout_ms_; }
    const std::map<std::string, std::string>& getOptions() const { return options_; }
    
    // Setters (fluent interface)
    ProcessorConfiguration& setName(const std::string& name) {
        name_ = name;
        return *this;
    }
    
    ProcessorConfiguration& setDebug(bool debug) {
        debug_ = debug;
        return *this;
    }
    
    ProcessorConfiguration& setTimeoutMs(int timeout) {
        timeout_ms_ = timeout;
        return *this;
    }
    
    ProcessorConfiguration& addOption(const std::string& key, const std::string& value) {
        options_[key] = value;
        return *this;
    }
    
    // Validation
    bool validate() const {
        return !name_.empty() && timeout_ms_ > 0;
    }
    
    // Static factory methods
    static ProcessorConfiguration createDefault(const std::string& name) {
        return ProcessorConfiguration(name);
    }
    
    static ProcessorConfiguration createDebug(const std::string& name) {
        return ProcessorConfiguration(name).setDebug(true);
    }
};

/**
 * Custom exception for processing errors
 */
class ProcessingException : public std::exception {
private:
    std::string message_;
    std::string item_;
    int error_code_;

public:
    enum ErrorCode {
        INVALID_INPUT = 1,
        PROCESSING_FAILED = 2,
        TIMEOUT = 3,
        RESOURCE_UNAVAILABLE = 4
    };
    
    ProcessingException(const std::string& message, ErrorCode code = PROCESSING_FAILED)
        : message_(message), error_code_(code) {}
    
    ProcessingException(const std::string& message, const std::string& item, ErrorCode code = PROCESSING_FAILED)
        : message_(message), item_(item), error_code_(code) {}
    
    const char* what() const noexcept override {
        return message_.c_str();
    }
    
    const std::string& getItem() const { return item_; }
    int getErrorCode() const { return error_code_; }
};

/**
 * Abstract base class for processors
 */
template<typename InputType, typename OutputType>
class BaseProcessor {
protected:
    ProcessorConfiguration config_;
    std::vector<OutputType> results_;
    std::atomic<int> error_count_;
    mutable std::mutex results_mutex_;
    
    // Protected virtual destructor for proper inheritance
    virtual ~BaseProcessor() = default;

public:
    explicit BaseProcessor(const ProcessorConfiguration& config)
        : config_(config), error_count_(0) {
        if (!config_.validate()) {
            throw ProcessingException("Invalid configuration", ProcessingException::INVALID_INPUT);
        }
    }
    
    // Pure virtual methods
    virtual OutputType processItem(const InputType& item) = 0;
    virtual bool validateItem(const InputType& item) const = 0;
    
    // Template method pattern
    std::vector<OutputType> process(const std::vector<InputType>& items) {
        std::vector<OutputType> processed;
        processed.reserve(items.size());
        
        for (const auto& item : items) {
            try {
                if (!validateItem(item)) {
                    throw ProcessingException("Invalid item", ProcessingException::INVALID_INPUT);
                }
                
                OutputType result = processItem(item);
                processed.push_back(result);
                addResult(result);
                
                if (config_.isDebug()) {
                    logDebug("Processed item successfully");
                }
                
            } catch (const ProcessingException& e) {
                error_count_++;
                handleError(e, item);
            }
        }
        
        return processed;
    }
    
    // Async processing with futures
    std::future<std::vector<OutputType>> processAsync(const std::vector<InputType>& items) {
        return std::async(std::launch::async, [this, items]() {
            return this->process(items);
        });
    }
    
    // Parallel processing
    std::vector<OutputType> processParallel(const std::vector<InputType>& items, 
                                          size_t num_threads = std::thread::hardware_concurrency()) {
        if (items.empty()) {
            return {};
        }
        
        std::vector<std::future<std::vector<OutputType>>> futures;
        size_t items_per_thread = items.size() / num_threads;
        size_t remaining = items.size() % num_threads;
        
        auto it = items.begin();
        
        for (size_t i = 0; i < num_threads; ++i) {
            size_t chunk_size = items_per_thread + (i < remaining ? 1 : 0);
            if (chunk_size == 0) break;
            
            auto end_it = it + chunk_size;
            std::vector<InputType> chunk(it, end_it);
            
            futures.push_back(std::async(std::launch::async, [this, chunk]() {
                return this->process(chunk);
            }));
            
            it = end_it;
        }
        
        // Combine results
        std::vector<OutputType> combined;
        for (auto& future : futures) {
            auto result = future.get();
            combined.insert(combined.end(), result.begin(), result.end());
        }
        
        return combined;
    }
    
    // Statistics
    std::map<std::string, int> getStats() const {
        std::lock_guard<std::mutex> lock(results_mutex_);
        return {
            {"total_processed", static_cast<int>(results_.size())},
            {"error_count", error_count_.load()},
            {"timeout_ms", config_.getTimeoutMs()}
        };
    }
    
    // Reset state
    void reset() {
        std::lock_guard<std::mutex> lock(results_mutex_);
        results_.clear();
        error_count_ = 0;
    }

protected:
    void addResult(const OutputType& result) {
        std::lock_guard<std::mutex> lock(results_mutex_);
        results_.push_back(result);
    }
    
    virtual void handleError(const ProcessingException& e, const InputType& item) {
        if (config_.isDebug()) {
            std::cerr << "Error processing item: " << e.what() << std::endl;
        }
    }
    
    void logDebug(const std::string& message) const {
        if (config_.isDebug()) {
            std::cout << "[DEBUG] " << config_.getName() << ": " << message << std::endl;
        }
    }
};

/**
 * Concrete string processor implementation
 */
class StringProcessor : public BaseProcessor<std::string, std::string> {
public:
    explicit StringProcessor(const ProcessorConfiguration& config)
        : BaseProcessor(config) {}
    
    std::string processItem(const std::string& item) override {
        // Simulate processing time
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        
        std::string result = "processed_";
        std::string lower_item = item;
        std::transform(lower_item.begin(), lower_item.end(), lower_item.begin(), ::tolower);
        
        // Remove whitespace
        lower_item.erase(std::remove_if(lower_item.begin(), lower_item.end(), ::isspace), lower_item.end());
        
        result += lower_item;
        
        logDebug("Processed: " + item + " -> " + result);
        return result;
    }
    
    bool validateItem(const std::string& item) const override {
        return !item.empty() && item.length() <= 1000;
    }
    
    // Custom processing with transform function
    template<typename TransformFunc>
    std::vector<std::string> processWithTransform(const std::vector<std::string>& items,
                                                 TransformFunc transform) {
        std::vector<std::string> results;
        results.reserve(items.size());
        
        for (const auto& item : items) {
            if (validateItem(item)) {
                try {
                    std::string processed = processItem(item);
                    results.push_back(transform(processed));
                } catch (const std::exception& e) {
                    error_count_++;
                    if (config_.isDebug()) {
                        std::cerr << "Transform error: " << e.what() << std::endl;
                    }
                }
            }
        }
        
        return results;
    }
};

/**
 * Generic batch processor template
 */
template<typename ProcessorType>
class BatchProcessor {
private:
    std::unique_ptr<ProcessorType> processor_;
    size_t batch_size_;

public:
    BatchProcessor(std::unique_ptr<ProcessorType> processor, size_t batch_size)
        : processor_(std::move(processor)), batch_size_(batch_size) {}
    
    template<typename InputType>
    auto processBatches(const std::vector<InputType>& items) 
        -> std::vector<decltype(processor_->processItem(items[0]))> {
        
        using OutputType = decltype(processor_->processItem(items[0]));
        std::vector<OutputType> all_results;
        
        for (size_t i = 0; i < items.size(); i += batch_size_) {
            size_t end = std::min(i + batch_size_, items.size());
            std::vector<InputType> batch(items.begin() + i, items.begin() + end);
            
            auto batch_results = processor_->process(batch);
            all_results.insert(all_results.end(), batch_results.begin(), batch_results.end());
        }
        
        return all_results;
    }
    
    ProcessorType* getProcessor() { return processor_.get(); }
    const ProcessorType* getProcessor() const { return processor_.get(); }
};

/**
 * RAII wrapper for processor operations
 */
template<typename ProcessorType>
class ProcessorGuard {
private:
    ProcessorType& processor_;
    std::string operation_name_;
    std::chrono::high_resolution_clock::time_point start_time_;

public:
    ProcessorGuard(ProcessorType& processor, const std::string& operation_name)
        : processor_(processor), operation_name_(operation_name),
          start_time_(std::chrono::high_resolution_clock::now()) {
        std::cout << "Starting operation: " << operation_name_ << std::endl;
    }
    
    ~ProcessorGuard() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_);
        std::cout << "Completed operation: " << operation_name_ 
                  << " in " << duration.count() << "ms" << std::endl;
    }
    
    // Non-copyable, non-movable
    ProcessorGuard(const ProcessorGuard&) = delete;
    ProcessorGuard& operator=(const ProcessorGuard&) = delete;
    ProcessorGuard(ProcessorGuard&&) = delete;
    ProcessorGuard& operator=(ProcessorGuard&&) = delete;
};

/**
 * Factory class for creating processors
 */
class ProcessorFactory {
public:
    static std::unique_ptr<StringProcessor> createStringProcessor(const std::string& name) {
        auto config = ProcessorConfiguration::createDefault(name);
        return std::make_unique<StringProcessor>(config);
    }
    
    static std::unique_ptr<StringProcessor> createDebugStringProcessor(const std::string& name) {
        auto config = ProcessorConfiguration::createDebug(name);
        return std::make_unique<StringProcessor>(config);
    }
    
    template<typename ProcessorType>
    static std::unique_ptr<BatchProcessor<ProcessorType>> createBatchProcessor(
        std::unique_ptr<ProcessorType> processor, size_t batch_size = 10) {
        return std::make_unique<BatchProcessor<ProcessorType>>(std::move(processor), batch_size);
    }
};

/**
 * Utility functions
 */
namespace Utils {
    std::vector<std::string> generateTestData(int count, const std::string& prefix = "item") {
        std::vector<std::string> items;
        items.reserve(count);
        
        for (int i = 0; i < count; ++i) {
            items.push_back(prefix + "_" + std::to_string(i));
        }
        
        return items;
    }
    
    bool validateItems(const std::vector<std::string>& items) {
        return !items.empty() && 
               std::all_of(items.begin(), items.end(), 
                          [](const std::string& item) { return !item.empty(); });
    }
    
    template<typename Container>
    void printResults(const Container& results, const std::string& title = "Results") {
        std::cout << title << " (" << results.size() << " items):" << std::endl;
        for (const auto& result : results) {
            std::cout << "  " << result << std::endl;
        }
    }
    
    // Function object for custom transformations
    class ToUpperTransform {
    public:
        std::string operator()(const std::string& input) const {
            std::string result = input;
            std::transform(result.begin(), result.end(), result.begin(), ::toupper);
            return result;
        }
    };
}

} // namespace ProcessorFramework

/**
 * Main function demonstrating C++ features
 */
int main() {
    using namespace ProcessorFramework;
    
    std::cout << "Starting C++ processor application..." << std::endl;
    
    try {
        // Create processor using factory
        auto processor = ProcessorFactory::createDebugStringProcessor("main");
        
        // Generate test data
        auto testData = Utils::generateTestData(5, "item");
        
        if (!Utils::validateItems(testData)) {
            throw ProcessingException("Invalid test data", ProcessingException::INVALID_INPUT);
        }
        
        // Synchronous processing with RAII guard
        {
            ProcessorGuard<StringProcessor> guard(*processor, "Synchronous Processing");
            auto results = processor->process(testData);
            Utils::printResults(results, "Synchronous Results");
        }
        
        // Asynchronous processing
        {
            ProcessorGuard<StringProcessor> guard(*processor, "Asynchronous Processing");
            auto future = processor->processAsync(testData);
            auto asyncResults = future.get();
            Utils::printResults(asyncResults, "Asynchronous Results");
        }
        
        // Parallel processing
        {
            ProcessorGuard<StringProcessor> guard(*processor, "Parallel Processing");
            auto parallelResults = processor->processParallel(testData, 2);
            Utils::printResults(parallelResults, "Parallel Results");
        }
        
        // Custom transform processing
        {
            ProcessorGuard<StringProcessor> guard(*processor, "Transform Processing");
            auto transformResults = processor->processWithTransform(testData, Utils::ToUpperTransform());
            Utils::printResults(transformResults, "Transform Results");
        }
        
        // Batch processing
        {
            ProcessorGuard<StringProcessor> guard(*processor, "Batch Processing");
            auto batchProcessor = ProcessorFactory::createBatchProcessor(std::move(processor), 2);
            auto batchResults = batchProcessor->processBatches(testData);
            Utils::printResults(batchResults, "Batch Results");
        }
        
        std::cout << "C++ processor application completed successfully." << std::endl;
        
    } catch (const ProcessingException& e) {
        std::cerr << "Processing error: " << e.what() << " (code: " << e.getErrorCode() << ")" << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}