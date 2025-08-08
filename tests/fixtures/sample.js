/**
 * JavaScript test fixture for Tree-sitter parsing tests.
 * 
 * This file contains various JavaScript language constructs to test comprehensive parsing.
 */

const fs = require('fs');
const path = require('path');
const { EventEmitter } = require('events');

// ES6 Class with various features
class DataProcessor extends EventEmitter {
    /**
     * Creates a new DataProcessor instance.
     * @param {Object} config - Configuration object
     * @param {string} config.name - Processor name
     * @param {boolean} config.debug - Debug mode flag
     */
    constructor(config = {}) {
        super();
        this.config = {
            name: 'default',
            debug: false,
            ...config
        };
        this.results = [];
        this.errorCount = 0;
        this._initialized = true;
        
        // Bind methods to preserve 'this' context
        this.process = this.process.bind(this);
        this.processAsync = this.processAsync.bind(this);
    }
    
    /**
     * Process array of data items synchronously.
     * @param {string[]} data - Array of strings to process
     * @returns {string[]} Processed data array
     */
    process(data) {
        if (!this.validateData(data)) {
            throw new Error('Invalid input data');
        }
        
        const processed = [];
        
        for (const item of data) {
            try {
                const result = this._processItem(item);
                processed.push(result);
                this.results.push(result);
                this.emit('itemProcessed', result);
            } catch (error) {
                this.errorCount++;
                this._handleError(error, item);
            }
        }
        
        return processed;
    }
    
    /**
     * Process data items asynchronously.
     * @param {string[]} data - Array of strings to process
     * @returns {Promise<string[]>} Promise resolving to processed data
     */
    async processAsync(data) {
        const results = [];
        
        // Process items in parallel using Promise.all
        const promises = data.map(item => this._processItemAsync(item));
        
        try {
            const processedItems = await Promise.all(promises);
            results.push(...processedItems);
            this.results.push(...processedItems);
            
            // Emit batch completion event
            this.emit('batchCompleted', results);
        } catch (error) {
            this.emit('error', error);
            throw error;
        }
        
        return results;
    }
    
    /**
     * Process single item (private method).
     * @private
     * @param {string} item - Item to process
     * @returns {string} Processed item
     */
    _processItem(item) {
        if (!item || typeof item !== 'string' || item.trim().length === 0) {
            throw new Error('Invalid item: empty or non-string');
        }
        
        return `processed_${item.trim().toLowerCase()}`;
    }
    
    /**
     * Process single item asynchronously (private method).
     * @private
     * @param {string} item - Item to process
     * @returns {Promise<string>} Promise resolving to processed item
     */
    async _processItemAsync(item) {
        // Simulate async work with setTimeout
        await new Promise(resolve => setTimeout(resolve, 1));
        return this._processItem(item);
    }
    
    /**
     * Handle processing errors.
     * @private
     * @param {Error} error - The error that occurred
     * @param {string} item - The item that caused the error
     */
    _handleError(error, item) {
        if (this.config.debug) {
            console.error(`Error processing '${item}':`, error.message);
        }
        this.emit('error', error, item);
    }
    
    /**
     * Get processing statistics.
     * @returns {Object} Statistics object
     */
    getStats() {
        return {
            totalProcessed: this.results.length,
            errorCount: this.errorCount,
            configName: this.config.name
        };
    }
    
    /**
     * Validate input data array.
     * @param {*} data - Data to validate
     * @returns {boolean} True if data is valid
     */
    validateData(data) {
        return Array.isArray(data) && data.every(item => typeof item === 'string');
    }
    
    /**
     * Reset processor state.
     */
    reset() {
        this.results = [];
        this.errorCount = 0;
        this.emit('reset');
    }
    
    /**
     * Create processor from configuration file.
     * @static
     * @param {string} configPath - Path to configuration file
     * @returns {DataProcessor} New processor instance
     */
    static fromConfigFile(configPath) {
        try {
            const configData = JSON.parse(fs.readFileSync(configPath, 'utf8'));
            return new DataProcessor(configData);
        } catch (error) {
            throw new Error(`Failed to load config from ${configPath}: ${error.message}`);
        }
    }
    
    /**
     * Get default configuration.
     * @static
     * @returns {Object} Default configuration object
     */
    static getDefaultConfig() {
        return {
            name: 'default',
            debug: false,
            batchSize: 100
        };
    }
}

// Arrow functions and modern JavaScript features
const createProcessor = (name = 'default', options = {}) => {
    const config = { name, ...DataProcessor.getDefaultConfig(), ...options };
    return new DataProcessor(config);
};

// Async arrow function
const batchProcess = async (items, processor) => {
    if (!processor.validateData(items)) {
        throw new Error('Invalid input data for batch processing');
    }
    
    const results = await processor.processAsync(items);
    const stats = processor.getStats();
    
    return {
        results,
        stats,
        timestamp: new Date().toISOString()
    };
};

// Generator function
function* generateTestData(count = 10) {
    for (let i = 0; i < count; i++) {
        yield `item_${i.toString().padStart(3, '0')}`;
    }
}

// Higher-order function with callback
function processWithCallback(data, processor, callback) {
    const results = processor.process(data);
    
    if (typeof callback === 'function') {
        results.forEach(result => callback(result));
    }
    
    return results;
}

// Promise-based function
function processWithPromise(data, processor) {
    return new Promise((resolve, reject) => {
        try {
            const results = processor.process(data);
            resolve(results);
        } catch (error) {
            reject(error);
        }
    });
}

// Class with static methods and getters/setters
class ConfigManager {
    constructor(initialConfig = {}) {
        this._config = { ...ConfigManager.getDefaults(), ...initialConfig };
        this._listeners = [];
    }
    
    get config() {
        return { ...this._config };
    }
    
    set config(newConfig) {
        const oldConfig = this._config;
        this._config = { ...this._config, ...newConfig };
        this._notifyListeners(oldConfig, this._config);
    }
    
    /**
     * Add configuration change listener.
     * @param {Function} listener - Listener function
     */
    addListener(listener) {
        if (typeof listener === 'function') {
            this._listeners.push(listener);
        }
    }
    
    /**
     * Remove configuration change listener.
     * @param {Function} listener - Listener function to remove
     */
    removeListener(listener) {
        const index = this._listeners.indexOf(listener);
        if (index > -1) {
            this._listeners.splice(index, 1);
        }
    }
    
    /**
     * Notify all listeners of configuration changes.
     * @private
     * @param {Object} oldConfig - Previous configuration
     * @param {Object} newConfig - New configuration
     */
    _notifyListeners(oldConfig, newConfig) {
        this._listeners.forEach(listener => {
            try {
                listener(oldConfig, newConfig);
            } catch (error) {
                console.error('Error in config listener:', error);
            }
        });
    }
    
    /**
     * Get default configuration values.
     * @static
     * @returns {Object} Default configuration
     */
    static getDefaults() {
        return {
            debug: false,
            timeout: 5000,
            retries: 3
        };
    }
}

// Destructuring and spread operator examples
const { name: processorName, debug } = DataProcessor.getDefaultConfig();
const extendedConfig = { ...DataProcessor.getDefaultConfig(), custom: true };

// Template literals and tagged templates
const formatMessage = (template, ...values) => {
    return template.reduce((result, string, index) => {
        return result + string + (values[index] || '');
    }, '');
};

const logMessage = (strings, ...values) => {
    const timestamp = new Date().toISOString();
    const message = strings.reduce((result, string, index) => {
        return result + string + (values[index] || '');
    }, '');
    console.log(`[${timestamp}] ${message}`);
};

// Export patterns (CommonJS and ES6 mixed for testing)
module.exports = {
    DataProcessor,
    ConfigManager,
    createProcessor,
    batchProcess,
    generateTestData,
    processWithCallback,
    processWithPromise,
    formatMessage,
    logMessage
};

// Default export simulation
module.exports.default = DataProcessor;

// Example usage and main execution
if (require.main === module) {
    // Self-executing async function
    (async () => {
        try {
            const testData = Array.from(generateTestData(5));
            const processor = createProcessor('main', { debug: true });
            
            // Add event listeners
            processor.on('itemProcessed', (result) => {
                console.log(`Item processed: ${result}`);
            });
            
            processor.on('error', (error, item) => {
                console.error(`Processing error for '${item}':`, error.message);
            });
            
            // Process data
            const results = await batchProcess(testData, processor);
            console.log('Batch processing results:', results);
            
            // Configuration management example
            const configManager = new ConfigManager();
            configManager.addListener((oldConfig, newConfig) => {
                console.log('Configuration changed:', { oldConfig, newConfig });
            });
            
            configManager.config = { debug: true, timeout: 10000 };
            
        } catch (error) {
            console.error('Main execution error:', error);
            process.exit(1);
        }
    })();
}