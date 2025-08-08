// Rust test fixture for Tree-sitter parsing tests.
//
// This file contains various Rust language constructs to test comprehensive parsing.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use serde::{Deserialize, Serialize};

/// Configuration struct with serialization support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfiguration {
    pub name: String,
    pub debug: bool,
    pub timeout_ms: u64,
    pub options: HashMap<String, String>,
}

impl ProcessorConfiguration {
    /// Create a new configuration with defaults
    pub fn new(name: String) -> Self {
        Self {
            name,
            debug: false,
            timeout_ms: 5000,
            options: HashMap::new(),
        }
    }
    
    /// Builder pattern for configuration
    pub fn builder() -> ConfigurationBuilder {
        ConfigurationBuilder::new()
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.name.is_empty() {
            return Err("Configuration name cannot be empty".into());
        }
        if self.timeout_ms == 0 {
            return Err("Timeout must be greater than zero".into());
        }
        Ok(())
    }
}

/// Builder for ProcessorConfiguration
pub struct ConfigurationBuilder {
    name: Option<String>,
    debug: bool,
    timeout_ms: u64,
    options: HashMap<String, String>,
}

impl ConfigurationBuilder {
    pub fn new() -> Self {
        Self {
            name: None,
            debug: false,
            timeout_ms: 5000,
            options: HashMap::new(),
        }
    }
    
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }
    
    pub fn debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }
    
    pub fn timeout_ms(mut self, timeout: u64) -> Self {
        self.timeout_ms = timeout;
        self
    }
    
    pub fn option<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.options.insert(key.into(), value.into());
        self
    }
    
    pub fn build(self) -> Result<ProcessorConfiguration, Box<dyn Error>> {
        let name = self.name.ok_or("Name is required")?;
        let config = ProcessorConfiguration {
            name,
            debug: self.debug,
            timeout_ms: self.timeout_ms,
            options: self.options,
        };
        config.validate()?;
        Ok(config)
    }
}

impl Default for ConfigurationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Custom error types for processing
#[derive(Debug)]
pub enum ProcessingError {
    InvalidInput(String),
    ProcessingFailed(String),
    Timeout,
    ResourceUnavailable,
}

impl fmt::Display for ProcessingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProcessingError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            ProcessingError::ProcessingFailed(msg) => write!(f, "Processing failed: {}", msg),
            ProcessingError::Timeout => write!(f, "Operation timed out"),
            ProcessingError::ResourceUnavailable => write!(f, "Resource unavailable"),
        }
    }
}

impl Error for ProcessingError {}

/// Result type alias for processing operations
pub type ProcessingResult<T> = Result<T, ProcessingError>;

/// Trait for data processing operations
pub trait Processor: Send + Sync {
    /// Process a single item
    fn process_item(&self, item: &str) -> ProcessingResult<String>;
    
    /// Process multiple items
    fn process(&self, items: &[String]) -> Vec<ProcessingResult<String>> {
        items.iter().map(|item| self.process_item(item)).collect()
    }
    
    /// Get processing statistics
    fn get_stats(&self) -> HashMap<String, u64>;
    
    /// Reset processor state
    fn reset(&mut self);
}

/// Async trait for asynchronous processing
#[async_trait::async_trait]
pub trait AsyncProcessor: Send + Sync {
    /// Process items asynchronously
    async fn process_async(&self, items: Vec<String>) -> Vec<ProcessingResult<String>>;
    
    /// Process items with streaming results
    async fn process_stream(&self, items: Vec<String>) -> mpsc::Receiver<ProcessingResult<String>>;
}

/// Concrete implementation of a data processor
pub struct DataProcessor {
    config: ProcessorConfiguration,
    results: Arc<Mutex<Vec<String>>>,
    error_count: Arc<Mutex<u64>>,
    stats: Arc<RwLock<HashMap<String, u64>>>,
}

impl DataProcessor {
    /// Create a new DataProcessor
    pub fn new(config: ProcessorConfiguration) -> ProcessingResult<Self> {
        config.validate().map_err(|e| {
            ProcessingError::InvalidInput(e.to_string())
        })?;
        
        let mut stats = HashMap::new();
        stats.insert("total_processed".to_string(), 0);
        stats.insert("error_count".to_string(), 0);
        
        Ok(Self {
            config,
            results: Arc::new(Mutex::new(Vec::new())),
            error_count: Arc::new(Mutex::new(0)),
            stats: Arc::new(RwLock::new(stats)),
        })
    }
    
    /// Create processor with default configuration
    pub fn with_default_config(name: String) -> ProcessingResult<Self> {
        let config = ProcessorConfiguration::new(name);
        Self::new(config)
    }
    
    /// Add result to internal storage
    fn add_result(&self, result: String) {
        if let Ok(mut results) = self.results.lock() {
            results.push(result);
        }
        
        // Update stats
        if let Ok(mut stats) = self.stats.write() {
            *stats.entry("total_processed".to_string()).or_insert(0) += 1;
        }
    }
    
    /// Increment error count
    fn increment_error_count(&self) {
        if let Ok(mut count) = self.error_count.lock() {
            *count += 1;
        }
        
        // Update stats
        if let Ok(mut stats) = self.stats.write() {
            *stats.entry("error_count".to_string()).or_insert(0) += 1;
        }
    }
    
    /// Validate input item
    fn validate_item(&self, item: &str) -> ProcessingResult<()> {
        if item.is_empty() {
            return Err(ProcessingError::InvalidInput("Empty item".to_string()));
        }
        if item.len() > 1000 {
            return Err(ProcessingError::InvalidInput("Item too long".to_string()));
        }
        Ok(())
    }
    
    /// Log debug message if debug mode is enabled
    fn log_debug(&self, message: &str) {
        if self.config.debug {
            println!("[DEBUG] {}: {}", self.config.name, message);
        }
    }
}

impl Processor for DataProcessor {
    fn process_item(&self, item: &str) -> ProcessingResult<String> {
        self.validate_item(item)?;
        
        // Simulate processing time
        thread::sleep(Duration::from_millis(1));
        
        let result = format!("processed_{}", item.trim().to_lowercase());
        
        self.log_debug(&format!("Processed: {} -> {}", item, result));
        self.add_result(result.clone());
        
        Ok(result)
    }
    
    fn get_stats(&self) -> HashMap<String, u64> {
        self.stats.read().unwrap().clone()
    }
    
    fn reset(&mut self) {
        if let Ok(mut results) = self.results.lock() {
            results.clear();
        }
        if let Ok(mut count) = self.error_count.lock() {
            *count = 0;
        }
        if let Ok(mut stats) = self.stats.write() {
            for value in stats.values_mut() {
                *value = 0;
            }
        }
    }
}

#[async_trait::async_trait]
impl AsyncProcessor for DataProcessor {
    async fn process_async(&self, items: Vec<String>) -> Vec<ProcessingResult<String>> {
        let mut handles = vec![];
        
        for item in items {
            let processor = Arc::new(self);
            let handle = tokio::spawn(async move {
                processor.process_item(&item)
            });
            handles.push(handle);
        }
        
        let mut results = vec![];
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => results.push(Err(ProcessingError::ProcessingFailed(e.to_string()))),
            }
        }
        
        results
    }
    
    async fn process_stream(&self, items: Vec<String>) -> mpsc::Receiver<ProcessingResult<String>> {
        let (tx, rx) = mpsc::channel(items.len());
        
        let processor = Arc::new(self);
        tokio::spawn(async move {
            for item in items {
                let result = processor.process_item(&item);
                if tx.send(result).await.is_err() {
                    break; // Receiver dropped
                }
            }
        });
        
        rx
    }
}

/// Batch processor for handling large datasets
pub struct BatchProcessor<P: Processor> {
    processor: P,
    batch_size: usize,
}

impl<P: Processor> BatchProcessor<P> {
    pub fn new(processor: P, batch_size: usize) -> Self {
        Self {
            processor,
            batch_size: batch_size.max(1), // Ensure batch size is at least 1
        }
    }
    
    pub fn process_batches(&self, items: Vec<String>) -> Vec<ProcessingResult<String>> {
        let mut all_results = Vec::new();
        
        for batch in items.chunks(self.batch_size) {
            let batch_vec = batch.to_vec();
            let mut batch_results = self.processor.process(&batch_vec);
            all_results.append(&mut batch_results);
        }
        
        all_results
    }
}

/// Generic worker for concurrent processing
pub struct Worker<T> {
    id: usize,
    receiver: mpsc::Receiver<T>,
}

impl<T> Worker<T> {
    pub fn new(id: usize, receiver: mpsc::Receiver<T>) -> Self {
        Self { id, receiver }
    }
    
    pub async fn run<F, R>(mut self, mut handler: F)
    where
        F: FnMut(usize, T) -> R + Send,
        T: Send,
        R: Send,
    {
        while let Some(item) = self.receiver.recv().await {
            let _result = handler(self.id, item);
        }
    }
}

/// Thread pool for processing
pub struct ProcessorPool {
    workers: Vec<thread::JoinHandle<()>>,
    sender: mpsc::Sender<String>,
}

impl ProcessorPool {
    pub fn new(size: usize, processor: Arc<dyn Processor>) -> Self {
        let (sender, mut receiver) = mpsc::channel::<String>(100);
        let mut workers = Vec::with_capacity(size);
        
        for id in 0..size {
            let processor_clone = Arc::clone(&processor);
            let mut receiver_clone = receiver.clone();
            
            let handle = thread::spawn(move || {
                tokio::runtime::Runtime::new().unwrap().block_on(async move {
                    while let Some(item) = receiver_clone.recv().await {
                        match processor_clone.process_item(&item) {
                            Ok(result) => println!("Worker {}: {}", id, result),
                            Err(e) => eprintln!("Worker {} error: {}", id, e),
                        }
                    }
                });
            });
            
            workers.push(handle);
        }
        
        Self { workers, sender }
    }
    
    pub async fn submit(&self, item: String) -> Result<(), ProcessingError> {
        self.sender
            .send(item)
            .await
            .map_err(|_| ProcessingError::ResourceUnavailable)
    }
    
    pub fn shutdown(self) {
        drop(self.sender); // Close the channel
        
        for worker in self.workers {
            worker.join().unwrap();
        }
    }
}

/// Utility functions
pub mod utils {
    use super::*;
    
    /// Create a processor with default configuration
    pub fn create_default_processor(name: &str) -> ProcessingResult<DataProcessor> {
        let config = ProcessorConfiguration::builder()
            .name(name)
            .debug(false)
            .timeout_ms(5000)
            .build()?;
        DataProcessor::new(config)
    }
    
    /// Generate test data
    pub fn generate_test_data(count: usize, prefix: &str) -> Vec<String> {
        (0..count)
            .map(|i| format!("{}_{:03}", prefix, i))
            .collect()
    }
    
    /// Validate all items in a collection
    pub fn validate_items(items: &[String]) -> ProcessingResult<()> {
        if items.is_empty() {
            return Err(ProcessingError::InvalidInput("No items provided".to_string()));
        }
        
        for (index, item) in items.iter().enumerate() {
            if item.is_empty() {
                return Err(ProcessingError::InvalidInput(
                    format!("Empty item at index {}", index)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Process items with timeout
    pub async fn process_with_timeout(
        processor: Arc<dyn AsyncProcessor>,
        items: Vec<String>,
        timeout: Duration,
    ) -> ProcessingResult<Vec<ProcessingResult<String>>> {
        let future = processor.process_async(items);
        
        match tokio::time::timeout(timeout, future).await {
            Ok(results) => Ok(results),
            Err(_) => Err(ProcessingError::Timeout),
        }
    }
}

/// Configuration management
pub mod config {
    use super::*;
    use std::fs;
    use std::path::Path;
    
    /// Save configuration to file
    pub fn save_config(config: &ProcessorConfiguration, path: &Path) -> Result<(), Box<dyn Error>> {
        let json = serde_json::to_string_pretty(config)?;
        fs::write(path, json)?;
        Ok(())
    }
    
    /// Load configuration from file
    pub fn load_config(path: &Path) -> Result<ProcessorConfiguration, Box<dyn Error>> {
        let content = fs::read_to_string(path)?;
        let config: ProcessorConfiguration = serde_json::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }
}

/// Example usage and main function
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("Starting Rust processor application...");
    
    // Create configuration
    let config = ProcessorConfiguration::builder()
        .name("main")
        .debug(true)
        .timeout_ms(10000)
        .option("key1", "value1")
        .build()?;
    
    // Create processor
    let mut processor = DataProcessor::new(config)?;
    
    // Generate test data
    let test_data = utils::generate_test_data(5, "item");
    
    // Validate items
    utils::validate_items(&test_data)?;
    
    // Process synchronously
    println!("Processing synchronously...");
    let results = processor.process(&test_data);
    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(value) => println!("Result {}: {}", i, value),
            Err(e) => println!("Error {}: {}", i, e),
        }
    }
    
    // Process asynchronously
    println!("Processing asynchronously...");
    let async_results = processor.process_async(test_data.clone()).await;
    println!("Async results count: {}", async_results.len());
    
    // Process with streaming
    println!("Processing with streaming...");
    let mut stream = processor.process_stream(test_data.clone()).await;
    while let Some(result) = stream.recv().await {
        match result {
            Ok(value) => println!("Stream result: {}", value),
            Err(e) => println!("Stream error: {}", e),
        }
    }
    
    // Get statistics
    let stats = processor.get_stats();
    println!("Statistics: {:?}", stats);
    
    // Test batch processing
    let batch_processor = BatchProcessor::new(processor, 2);
    let batch_results = batch_processor.process_batches(test_data);
    println!("Batch results count: {}", batch_results.len());
    
    println!("Rust processor application completed successfully.");
    Ok(())
}