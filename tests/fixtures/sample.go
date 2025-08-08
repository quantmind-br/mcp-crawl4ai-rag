// Go test fixture for Tree-sitter parsing tests.
//
// This file contains various Go language constructs to test comprehensive parsing.

package fixtures

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"sync"
	"time"
)

// Configuration struct with JSON tags
type Configuration struct {
	Name    string            `json:"name"`
	Debug   bool              `json:"debug"`
	Timeout time.Duration     `json:"timeout"`
	Options map[string]string `json:"options,omitempty"`
}

// NewConfiguration creates a new configuration with defaults
func NewConfiguration(name string) *Configuration {
	return &Configuration{
		Name:    name,
		Debug:   false,
		Timeout: 5 * time.Second,
		Options: make(map[string]string),
	}
}

// Validate validates the configuration
func (c *Configuration) Validate() error {
	if c.Name == "" {
		return errors.New("configuration name cannot be empty")
	}
	if c.Timeout <= 0 {
		return errors.New("timeout must be positive")
	}
	return nil
}

// ToJSON converts configuration to JSON
func (c *Configuration) ToJSON() ([]byte, error) {
	return json.Marshal(c)
}

// FromJSON loads configuration from JSON
func (c *Configuration) FromJSON(data []byte) error {
	return json.Unmarshal(data, c)
}

// Processor interface defines processing operations
type Processor interface {
	Process(ctx context.Context, items []string) ([]string, error)
	ProcessAsync(ctx context.Context, items []string) (<-chan string, <-chan error)
	GetStats() map[string]interface{}
	Close() error
}

// DataProcessor implements the Processor interface
type DataProcessor struct {
	config     *Configuration
	results    []string
	errorCount int
	mutex      sync.RWMutex
	done       chan struct{}
	wg         sync.WaitGroup
}

// NewDataProcessor creates a new DataProcessor
func NewDataProcessor(config *Configuration) (*DataProcessor, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}
	
	return &DataProcessor{
		config:  config,
		results: make([]string, 0),
		done:    make(chan struct{}),
	}, nil
}

// Process processes items synchronously
func (dp *DataProcessor) Process(ctx context.Context, items []string) ([]string, error) {
	if len(items) == 0 {
		return nil, errors.New("no items to process")
	}
	
	processed := make([]string, 0, len(items))
	
	for _, item := range items {
		select {
		case <-ctx.Done():
			return processed, ctx.Err()
		default:
		}
		
		result, err := dp.processItem(item)
		if err != nil {
			dp.incrementErrorCount()
			if dp.config.Debug {
				log.Printf("Error processing '%s': %v", item, err)
			}
			continue
		}
		
		processed = append(processed, result)
		dp.addResult(result)
		
		if dp.config.Debug {
			log.Printf("Processed: %s -> %s", item, result)
		}
	}
	
	return processed, nil
}

// ProcessAsync processes items asynchronously
func (dp *DataProcessor) ProcessAsync(ctx context.Context, items []string) (<-chan string, <-chan error) {
	results := make(chan string, len(items))
	errors := make(chan error, len(items))
	
	go func() {
		defer close(results)
		defer close(errors)
		
		var wg sync.WaitGroup
		semaphore := make(chan struct{}, 10) // Limit concurrency to 10
		
		for _, item := range items {
			select {
			case <-ctx.Done():
				errors <- ctx.Err()
				return
			case semaphore <- struct{}{}:
			}
			
			wg.Add(1)
			go func(item string) {
				defer func() {
					<-semaphore
					wg.Done()
				}()
				
				result, err := dp.processItem(item)
				if err != nil {
					dp.incrementErrorCount()
					errors <- fmt.Errorf("processing '%s': %w", item, err)
					return
				}
				
				dp.addResult(result)
				results <- result
			}(item)
		}
		
		wg.Wait()
	}()
	
	return results, errors
}

// processItem processes a single item (private method)
func (dp *DataProcessor) processItem(item string) (string, error) {
	if item == "" {
		return "", errors.New("empty item")
	}
	
	// Simulate processing time
	time.Sleep(1 * time.Millisecond)
	
	return fmt.Sprintf("processed_%s", item), nil
}

// addResult adds a result to the processor state (thread-safe)
func (dp *DataProcessor) addResult(result string) {
	dp.mutex.Lock()
	defer dp.mutex.Unlock()
	dp.results = append(dp.results, result)
}

// incrementErrorCount increments the error count (thread-safe)
func (dp *DataProcessor) incrementErrorCount() {
	dp.mutex.Lock()
	defer dp.mutex.Unlock()
	dp.errorCount++
}

// GetStats returns processing statistics
func (dp *DataProcessor) GetStats() map[string]interface{} {
	dp.mutex.RLock()
	defer dp.mutex.RUnlock()
	
	return map[string]interface{}{
		"totalProcessed": len(dp.results),
		"errorCount":     dp.errorCount,
		"configName":     dp.config.Name,
	}
}

// Close closes the processor and cleans up resources
func (dp *DataProcessor) Close() error {
	close(dp.done)
	dp.wg.Wait()
	
	if dp.config.Debug {
		log.Printf("Processor '%s' closed", dp.config.Name)
	}
	
	return nil
}

// BatchProcessor processes items in batches
type BatchProcessor struct {
	processor *DataProcessor
	batchSize int
}

// NewBatchProcessor creates a new BatchProcessor
func NewBatchProcessor(processor *DataProcessor, batchSize int) *BatchProcessor {
	if batchSize <= 0 {
		batchSize = 10 // default batch size
	}
	
	return &BatchProcessor{
		processor: processor,
		batchSize: batchSize,
	}
}

// ProcessBatches processes items in batches
func (bp *BatchProcessor) ProcessBatches(ctx context.Context, items []string) ([]string, error) {
	var allResults []string
	
	for i := 0; i < len(items); i += bp.batchSize {
		end := i + bp.batchSize
		if end > len(items) {
			end = len(items)
		}
		
		batch := items[i:end]
		results, err := bp.processor.Process(ctx, batch)
		if err != nil {
			return allResults, fmt.Errorf("batch processing failed: %w", err)
		}
		
		allResults = append(allResults, results...)
	}
	
	return allResults, nil
}

// ProcessorManager manages multiple processors
type ProcessorManager struct {
	processors map[string]Processor
	mutex      sync.RWMutex
}

// NewProcessorManager creates a new ProcessorManager
func NewProcessorManager() *ProcessorManager {
	return &ProcessorManager{
		processors: make(map[string]Processor),
	}
}

// AddProcessor adds a processor to the manager
func (pm *ProcessorManager) AddProcessor(name string, processor Processor) {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()
	pm.processors[name] = processor
}

// GetProcessor retrieves a processor by name
func (pm *ProcessorManager) GetProcessor(name string) (Processor, bool) {
	pm.mutex.RLock()
	defer pm.mutex.RUnlock()
	processor, exists := pm.processors[name]
	return processor, exists
}

// RemoveProcessor removes a processor from the manager
func (pm *ProcessorManager) RemoveProcessor(name string) error {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()
	
	if processor, exists := pm.processors[name]; exists {
		err := processor.Close()
		delete(pm.processors, name)
		return err
	}
	
	return errors.New("processor not found")
}

// ProcessWithAll processes items with all managed processors
func (pm *ProcessorManager) ProcessWithAll(ctx context.Context, items []string) map[string][]string {
	pm.mutex.RLock()
	defer pm.mutex.RUnlock()
	
	results := make(map[string][]string)
	var wg sync.WaitGroup
	var mutex sync.Mutex
	
	for name, processor := range pm.processors {
		wg.Add(1)
		go func(name string, processor Processor) {
			defer wg.Done()
			
			result, err := processor.Process(ctx, items)
			if err != nil {
				log.Printf("Error processing with %s: %v", name, err)
				return
			}
			
			mutex.Lock()
			results[name] = result
			mutex.Unlock()
		}(name, processor)
	}
	
	wg.Wait()
	return results
}

// Close closes all managed processors
func (pm *ProcessorManager) Close() error {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()
	
	var errors []error
	
	for name, processor := range pm.processors {
		if err := processor.Close(); err != nil {
			errors = append(errors, fmt.Errorf("closing processor %s: %w", name, err))
		}
	}
	
	if len(errors) > 0 {
		return fmt.Errorf("errors closing processors: %v", errors)
	}
	
	return nil
}

// Utility functions

// CreateDefaultProcessor creates a processor with default configuration
func CreateDefaultProcessor(name string) (*DataProcessor, error) {
	config := NewConfiguration(name)
	return NewDataProcessor(config)
}

// ValidateItems validates a slice of items
func ValidateItems(items []string) error {
	if len(items) == 0 {
		return errors.New("no items provided")
	}
	
	for i, item := range items {
		if item == "" {
			return fmt.Errorf("empty item at index %d", i)
		}
	}
	
	return nil
}

// GenerateTestData generates test data for processing
func GenerateTestData(count int, prefix string) []string {
	items := make([]string, count)
	for i := 0; i < count; i++ {
		items[i] = fmt.Sprintf("%s_%03d", prefix, i)
	}
	return items
}

// ProcessWithTimeout processes items with a timeout context
func ProcessWithTimeout(processor Processor, items []string, timeout time.Duration) ([]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	
	return processor.Process(ctx, items)
}

// Worker represents a generic worker
type Worker struct {
	ID       int
	JobQueue chan Job
	quit     chan bool
}

// Job represents work to be done
type Job struct {
	ID   int
	Data string
}

// NewWorker creates a new worker
func NewWorker(id int, jobQueue chan Job) *Worker {
	return &Worker{
		ID:       id,
		JobQueue: jobQueue,
		quit:     make(chan bool),
	}
}

// Start starts the worker
func (w *Worker) Start() {
	go func() {
		for {
			select {
			case job := <-w.JobQueue:
				// Process job
				result := fmt.Sprintf("worker_%d_processed_%s", w.ID, job.Data)
				log.Printf("Worker %d processed job %d: %s", w.ID, job.ID, result)
			case <-w.quit:
				return
			}
		}
	}()
}

// Stop stops the worker
func (w *Worker) Stop() {
	close(w.quit)
}

// Example usage and main function
func main() {
	fmt.Println("Starting Go processor application...")
	
	// Create configuration
	config := NewConfiguration("main")
	config.Debug = true
	config.Timeout = 10 * time.Second
	
	// Create processor
	processor, err := NewDataProcessor(config)
	if err != nil {
		log.Fatalf("Failed to create processor: %v", err)
	}
	defer processor.Close()
	
	// Generate test data
	testData := GenerateTestData(5, "item")
	
	// Validate items
	if err := ValidateItems(testData); err != nil {
		log.Fatalf("Invalid test data: %v", err)
	}
	
	// Process synchronously
	fmt.Println("Processing synchronously...")
	ctx := context.Background()
	results, err := processor.Process(ctx, testData)
	if err != nil {
		log.Printf("Processing error: %v", err)
	} else {
		fmt.Printf("Results: %v\n", results)
	}
	
	// Process asynchronously
	fmt.Println("Processing asynchronously...")
	resultChan, errorChan := processor.ProcessAsync(ctx, testData)
	
	var asyncResults []string
	done := make(chan bool)
	
	go func() {
		for {
			select {
			case result, ok := <-resultChan:
				if !ok {
					done <- true
					return
				}
				asyncResults = append(asyncResults, result)
			case err := <-errorChan:
				log.Printf("Async processing error: %v", err)
			}
		}
	}()
	
	<-done
	fmt.Printf("Async results: %v\n", asyncResults)
	
	// Get statistics
	stats := processor.GetStats()
	fmt.Printf("Statistics: %+v\n", stats)
	
	// Test batch processing
	batchProcessor := NewBatchProcessor(processor, 2)
	batchResults, err := batchProcessor.ProcessBatches(ctx, testData)
	if err != nil {
		log.Printf("Batch processing error: %v", err)
	} else {
		fmt.Printf("Batch results: %v\n", batchResults)
	}
	
	fmt.Println("Go processor application completed successfully.")
}