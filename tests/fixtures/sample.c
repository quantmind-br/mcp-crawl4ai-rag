/**
 * C test fixture for Tree-sitter parsing tests.
 * 
 * This file contains various C language constructs to test comprehensive parsing.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <pthread.h>
#include <unistd.h>

// Preprocessor definitions
#define MAX_ITEMS 100
#define BUFFER_SIZE 1024
#define VERSION "1.0.0"

// Conditional compilation
#ifdef DEBUG
#define LOG(msg) printf("[DEBUG] %s\n", msg)
#else
#define LOG(msg)
#endif

// Type definitions
typedef struct {
    char name[256];
    bool debug;
    int timeout_ms;
    void *options;
} ProcessorConfig;

typedef struct {
    char items[MAX_ITEMS][256];
    int count;
    int capacity;
    int error_count;
    pthread_mutex_t mutex;
} ProcessorResults;

// Function pointer types
typedef int (*ProcessorFunc)(const char *item, char *result, size_t result_size);
typedef void (*ErrorHandler)(const char *error_msg, const char *item);

// Global variables
static ProcessorResults g_results = {0};
static bool g_initialized = false;
static ErrorHandler g_error_handler = NULL;

// Forward declarations
int process_item(const char *item, char *result, size_t result_size);
int process_items(const char **items, int count, char results[][256]);
void *process_async_worker(void *arg);
void handle_error(const char *error_msg, const char *item);
bool validate_item(const char *item);

/**
 * Initialize the processor with configuration
 * @param config Processor configuration
 * @return 0 on success, -1 on error
 */
int processor_init(const ProcessorConfig *config) {
    if (!config) {
        return -1;
    }
    
    if (pthread_mutex_init(&g_results.mutex, NULL) != 0) {
        return -1;
    }
    
    g_results.count = 0;
    g_results.capacity = MAX_ITEMS;
    g_results.error_count = 0;
    g_initialized = true;
    
    LOG("Processor initialized");
    return 0;
}

/**
 * Create default processor configuration
 * @param name Processor name
 * @return Allocated configuration structure (must be freed)
 */
ProcessorConfig* create_default_config(const char *name) {
    ProcessorConfig *config = malloc(sizeof(ProcessorConfig));
    if (!config) {
        return NULL;
    }
    
    strncpy(config->name, name ? name : "default", sizeof(config->name) - 1);
    config->name[sizeof(config->name) - 1] = '\0';
    config->debug = false;
    config->timeout_ms = 5000;
    config->options = NULL;
    
    return config;
}

/**
 * Free processor configuration
 * @param config Configuration to free
 */
void free_config(ProcessorConfig *config) {
    if (config) {
        if (config->options) {
            free(config->options);
        }
        free(config);
    }
}

/**
 * Validate input item
 * @param item Item to validate
 * @return true if valid, false otherwise
 */
bool validate_item(const char *item) {
    if (!item) {
        return false;
    }
    
    size_t len = strlen(item);
    if (len == 0 || len > 255) {
        return false;
    }
    
    // Check for valid characters (alphanumeric and underscore)
    for (size_t i = 0; i < len; i++) {
        char c = item[i];
        if (!((c >= 'a' && c <= 'z') || 
              (c >= 'A' && c <= 'Z') || 
              (c >= '0' && c <= '9') || 
              c == '_' || c == ' ')) {
            return false;
        }
    }
    
    return true;
}

/**
 * Process a single item
 * @param item Input item to process
 * @param result Buffer for result
 * @param result_size Size of result buffer
 * @return 0 on success, -1 on error
 */
int process_item(const char *item, char *result, size_t result_size) {
    if (!item || !result) {
        return -1;
    }
    
    if (!validate_item(item)) {
        if (g_error_handler) {
            g_error_handler("Invalid item", item);
        }
        return -1;
    }
    
    // Process the item (convert to lowercase and add prefix)
    const char *prefix = "processed_";
    size_t prefix_len = strlen(prefix);
    size_t item_len = strlen(item);
    
    if (prefix_len + item_len + 1 > result_size) {
        if (g_error_handler) {
            g_error_handler("Result buffer too small", item);
        }
        return -1;
    }
    
    strcpy(result, prefix);
    
    // Convert to lowercase and append
    for (size_t i = 0; i < item_len; i++) {
        char c = item[i];
        if (c >= 'A' && c <= 'Z') {
            c = c - 'A' + 'a';
        }
        result[prefix_len + i] = c;
    }
    result[prefix_len + item_len] = '\0';
    
    return 0;
}

/**
 * Add result to global results storage
 * @param result Result to add
 * @return 0 on success, -1 on error
 */
int add_result(const char *result) {
    if (!g_initialized) {
        return -1;
    }
    
    pthread_mutex_lock(&g_results.mutex);
    
    if (g_results.count >= g_results.capacity) {
        pthread_mutex_unlock(&g_results.mutex);
        return -1;
    }
    
    strncpy(g_results.items[g_results.count], result, 255);
    g_results.items[g_results.count][255] = '\0';
    g_results.count++;
    
    pthread_mutex_unlock(&g_results.mutex);
    return 0;
}

/**
 * Process multiple items synchronously
 * @param items Array of items to process
 * @param count Number of items
 * @param results Array to store results
 * @return Number of successfully processed items
 */
int process_items(const char **items, int count, char results[][256]) {
    if (!items || !results || count <= 0) {
        return -1;
    }
    
    int processed_count = 0;
    
    for (int i = 0; i < count; i++) {
        char result[256];
        
        if (process_item(items[i], result, sizeof(result)) == 0) {
            strcpy(results[processed_count], result);
            add_result(result);
            processed_count++;
            
            LOG("Item processed successfully");
        } else {
            pthread_mutex_lock(&g_results.mutex);
            g_results.error_count++;
            pthread_mutex_unlock(&g_results.mutex);
        }
    }
    
    return processed_count;
}

/**
 * Structure for passing data to async worker threads
 */
typedef struct {
    const char **items;
    int start_index;
    int end_index;
    char (*results)[256];
    int *processed_count;
    pthread_mutex_t *count_mutex;
} AsyncWorkerArgs;

/**
 * Worker function for asynchronous processing
 * @param arg Pointer to AsyncWorkerArgs structure
 * @return NULL
 */
void *process_async_worker(void *arg) {
    AsyncWorkerArgs *args = (AsyncWorkerArgs *)arg;
    int local_count = 0;
    
    for (int i = args->start_index; i < args->end_index; i++) {
        char result[256];
        
        if (process_item(args->items[i], result, sizeof(result)) == 0) {
            pthread_mutex_lock(args->count_mutex);
            strcpy(args->results[*args->processed_count + local_count], result);
            local_count++;
            pthread_mutex_unlock(args->count_mutex);
            
            add_result(result);
            
            // Simulate processing time
            usleep(1000); // 1ms
        }
    }
    
    pthread_mutex_lock(args->count_mutex);
    *args->processed_count += local_count;
    pthread_mutex_unlock(args->count_mutex);
    
    return NULL;
}

/**
 * Process items asynchronously using multiple threads
 * @param items Array of items to process
 * @param count Number of items
 * @param results Array to store results
 * @param thread_count Number of threads to use
 * @return Number of successfully processed items
 */
int process_items_async(const char **items, int count, char results[][256], int thread_count) {
    if (!items || !results || count <= 0 || thread_count <= 0) {
        return -1;
    }
    
    pthread_t *threads = malloc(thread_count * sizeof(pthread_t));
    AsyncWorkerArgs *args = malloc(thread_count * sizeof(AsyncWorkerArgs));
    pthread_mutex_t count_mutex = PTHREAD_MUTEX_INITIALIZER;
    int processed_count = 0;
    
    if (!threads || !args) {
        free(threads);
        free(args);
        return -1;
    }
    
    int items_per_thread = count / thread_count;
    int remaining_items = count % thread_count;
    
    // Create worker threads
    for (int i = 0; i < thread_count; i++) {
        args[i].items = items;
        args[i].start_index = i * items_per_thread;
        args[i].end_index = args[i].start_index + items_per_thread;
        
        // Distribute remaining items to first few threads
        if (i < remaining_items) {
            args[i].end_index++;
        }
        
        args[i].results = results;
        args[i].processed_count = &processed_count;
        args[i].count_mutex = &count_mutex;
        
        if (pthread_create(&threads[i], NULL, process_async_worker, &args[i]) != 0) {
            // Handle thread creation error
            for (int j = 0; j < i; j++) {
                pthread_join(threads[j], NULL);
            }
            free(threads);
            free(args);
            pthread_mutex_destroy(&count_mutex);
            return -1;
        }
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < thread_count; i++) {
        pthread_join(threads[i], NULL);
    }
    
    free(threads);
    free(args);
    pthread_mutex_destroy(&count_mutex);
    
    return processed_count;
}

/**
 * Get processing statistics
 * @param total_processed Pointer to store total processed count
 * @param error_count Pointer to store error count
 */
void get_stats(int *total_processed, int *error_count) {
    if (!g_initialized) {
        if (total_processed) *total_processed = 0;
        if (error_count) *error_count = 0;
        return;
    }
    
    pthread_mutex_lock(&g_results.mutex);
    if (total_processed) *total_processed = g_results.count;
    if (error_count) *error_count = g_results.error_count;
    pthread_mutex_unlock(&g_results.mutex);
}

/**
 * Reset processor state
 */
void reset_processor(void) {
    if (!g_initialized) {
        return;
    }
    
    pthread_mutex_lock(&g_results.mutex);
    g_results.count = 0;
    g_results.error_count = 0;
    memset(g_results.items, 0, sizeof(g_results.items));
    pthread_mutex_unlock(&g_results.mutex);
}

/**
 * Set error handler callback
 * @param handler Error handler function
 */
void set_error_handler(ErrorHandler handler) {
    g_error_handler = handler;
}

/**
 * Default error handler implementation
 * @param error_msg Error message
 * @param item Item that caused the error
 */
void handle_error(const char *error_msg, const char *item) {
    fprintf(stderr, "Error processing '%s': %s\n", item ? item : "NULL", error_msg);
}

/**
 * Cleanup processor resources
 */
void processor_cleanup(void) {
    if (g_initialized) {
        pthread_mutex_destroy(&g_results.mutex);
        g_initialized = false;
        LOG("Processor cleaned up");
    }
}

/**
 * Generate test data
 * @param items Array to store generated items
 * @param count Number of items to generate
 * @param prefix Prefix for generated items
 */
void generate_test_data(char items[][256], int count, const char *prefix) {
    for (int i = 0; i < count; i++) {
        snprintf(items[i], 256, "%s_%03d", prefix ? prefix : "item", i);
    }
}

/**
 * Main function for testing
 */
int main(int argc, char *argv[]) {
    printf("Starting C processor application...\n");
    
    // Create configuration
    ProcessorConfig *config = create_default_config("main");
    if (!config) {
        fprintf(stderr, "Failed to create configuration\n");
        return 1;
    }
    
    config->debug = true;
    
    // Initialize processor
    if (processor_init(config) != 0) {
        fprintf(stderr, "Failed to initialize processor\n");
        free_config(config);
        return 1;
    }
    
    // Set error handler
    set_error_handler(handle_error);
    
    // Generate test data
    char test_items[5][256];
    generate_test_data(test_items, 5, "item");
    
    // Create array of pointers for processing
    const char *item_ptrs[5];
    for (int i = 0; i < 5; i++) {
        item_ptrs[i] = test_items[i];
    }
    
    // Process synchronously
    printf("Processing synchronously...\n");
    char sync_results[5][256];
    int sync_count = process_items(item_ptrs, 5, sync_results);
    
    printf("Synchronous results (%d items):\n", sync_count);
    for (int i = 0; i < sync_count; i++) {
        printf("  %s\n", sync_results[i]);
    }
    
    // Reset for async processing
    reset_processor();
    
    // Process asynchronously
    printf("Processing asynchronously...\n");
    char async_results[5][256];
    int async_count = process_items_async(item_ptrs, 5, async_results, 2);
    
    printf("Asynchronous results (%d items):\n", async_count);
    for (int i = 0; i < async_count; i++) {
        printf("  %s\n", async_results[i]);
    }
    
    // Get statistics
    int total_processed, error_count;
    get_stats(&total_processed, &error_count);
    printf("Statistics: Processed=%d, Errors=%d\n", total_processed, error_count);
    
    // Cleanup
    processor_cleanup();
    free_config(config);
    
    printf("C processor application completed successfully.\n");
    return 0;
}