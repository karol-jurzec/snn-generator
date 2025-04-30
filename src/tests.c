#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "../include/tests.h"
#include "../include/network.h"

#include "../include/models/model_base.h"
#include "../include/models/lif_neuron.h"

#include "../include/utils/snn_plot.h"
#include "../include/utils/layer_utils.h"
#include "../include/utils/network_loader.h"
#include "../include/utils/network_logger.h"
#include "../include/utils/nmnist_loader.h"

#include "../include/layers/conv2d_layer.h"
#include "../include/layers/maxpool2d_layer.h"
#include "../include/layers/flatten_layer.h"
#include "../include/layers/linear_layer.h"
#include "../include/layers/spiking_layer.h"


// Print the output feature map for inspection
void print_output(float *output, size_t out_channels, size_t output_dim) {
    for (size_t oc = 0; oc < out_channels; oc++) {
        printf("Output Channel %zu:\n", oc);
        for (size_t y = 0; y < output_dim; y++) {
            for (size_t x = 0; x < output_dim; x++) {
                printf("%0.2f ", output[(oc * output_dim * output_dim) + (y * output_dim + x)]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void print_output_spikes(float *output, size_t num_neurons) {
    printf("Output Spikes:\n");
    for (size_t i = 0; i < num_neurons; i++) {
        printf("Neuron %lu: Spiked = %0.1f\n", i, output[i]);
    }
    printf("\n");
}

void generate_synthetic_input(float *input, size_t size) {
    for (size_t i = 0; i < size; i++) {
        input[i] = (float)(i % 255) / 255.0f; // Normalize pixel values to [0, 1]
    }
}

void single_neuron_test(ModelBase *model_base, const char* filename) {
    int dt = 200;
    FILE *log_file = fopen("out/single_neuron_output.txt", "w");
    if (log_file == NULL) {
        perror("Error opening log file");
        return;
    }

        double arr[dt];
    for (int i = 0; i < dt; ++i) {
        if (i < 100) {
            arr[i] = 0.0;
        } else if (i < 150) {
            arr[i] = 0.25;
        } else {
            arr[i] = 0.0;
        }
    }

    for (int i = 0; i < dt; ++i) {
        model_base->update_neuron(model_base, arr[i]);  // Polymorphic call
        fprintf(log_file, "%d %f\n", i, model_base->v); // Log time and potential
    }

    fclose(log_file);
    plot_single_neuron("out/single_neuron_output.txt", filename);
}

void conv2d_test() {
    // Parameters
    int in_channels = 1; // Single channel input (e.g., grayscale image)
    int out_channels = 2; // Two output channels for test
    int kernel_size = 3;  // 3x3 kernel
    int stride = 1;       // Stride of 1
    int padding = 1;      // Padding of 1 (to preserve input size)

    size_t input_size = 14 * 14; // Assume 28x28 input image (MNIST-like)
    float *input = (float *)malloc(input_size * sizeof(float));

    for (size_t i = 0; i < input_size; i++) {
        //input[i] = (float)(i % 14 + i % 14) / (float)(14 + 14); // Simple pattern
        input[i] = 1.0f;
    }

    // Initialize Conv2D layer
    Conv2DLayer conv_layer;
    conv2d_initialize(&conv_layer, in_channels, out_channels, kernel_size, stride, padding, 28);

    // Perform forward pass
    conv2d_forward(&conv_layer, input, input_size);

    // Calculate output dimensions
    size_t output_dim = calculate_output_dim(14, kernel_size, stride, padding);

    // Print output feature map
    print_output(conv_layer.output, out_channels, output_dim);

    // Free resources
    free(input);
    conv2d_free(&conv_layer);
}

void maxpool2d_test() {
    int kernel_size = 2;
    int stride = 2;
    int padding = 0;

    size_t input_dim = 14;
    size_t input_size = input_dim * input_dim * 1; // 1 channel input
    float *input = (float *)malloc(input_size * sizeof(float));

    for (size_t i = 0; i < input_size; i++) {
        //input[i] = (float)(i % 14 + i % 14) / (float)(14 + 14);
        input[i] = 1.0f;
    }

    MaxPool2DLayer pool_layer;
    maxpool2d_initialize(&pool_layer, kernel_size, stride, padding, 28, 2);
    maxpool2d_forward(&pool_layer, input, input_size); 

    size_t output_dim = calculate_output_dim(input_dim, kernel_size, stride, padding);
    print_output(pool_layer.output, 1, output_dim);

    free(input);
    maxpool2d_free(&pool_layer);
}

void flatten_test() {
    size_t input_size = 4 * 4 * 2; // Example: 4x4 image with 2 channels (32 elements)
    float *input = (float *)malloc(input_size * sizeof(float));

    for (size_t i = 0; i < input_size; i++) {
        input[i] = (float)i / input_size; // Simple pattern
    }

    FlattenLayer flatten_layer;
    flatten_initialize(&flatten_layer, input_size);

    flatten_forward(&flatten_layer, input, input_size);

    for (size_t i = 0; i < flatten_layer.output_size; i++) {
        printf("%0.2f ", flatten_layer.output[i]);
    }
    printf("\n");

    free(input);
    flatten_free(&flatten_layer);
}

void linear_test() {
    size_t in_features = 16;  // Example input size
    size_t out_features = 4;  // Example output size
    float *input = (float *)malloc(in_features * sizeof(float));

    for (size_t i = 0; i < in_features; i++) {
        input[i] = (float)i / in_features;
    }

    LinearLayer linear_layer;
    linear_initialize(&linear_layer, in_features, out_features);

    linear_forward(&linear_layer, input, in_features);

    for (size_t i = 0; i < out_features; i++) {
        printf("%0.2f ", linear_layer.output[i]);
    }
    printf("\n");

    free(input);
    linear_free(&linear_layer);
}

void spiking_layer_test() {
    size_t num_neurons = 5;
    float *input = (float *)malloc(num_neurons * sizeof(float));
    for (size_t i = 0; i < num_neurons; i++) {
        input[i] = (float)i / num_neurons;
    }

    // Create neuron models (using LeakyLIF as an example)
    ModelBase *neuron_models[num_neurons];
    for (size_t i = 0; i < num_neurons; i++) {
        neuron_models[i] = (ModelBase *)malloc(sizeof(LIFNeuron));
        lif_initialize((LIFNeuron *)neuron_models[i], 0.0f, 1.0f, 0.0f, 0.5f);
    }

    // Initialize spiking layer
    SpikingLayer spiking_layer;
    spiking_initialize(&spiking_layer, num_neurons, neuron_models);

    // Perform forward pass
    spiking_forward(&spiking_layer, input, num_neurons);
    spiking_forward(&spiking_layer, input, num_neurons);

    // Print spike outputs
    printf("Spike Outputs:\n");
    for (size_t i = 0; i < num_neurons; i++) {
        printf("Neuron %lu: Spiked = %0.1f\n", i, spiking_layer.output_spikes[i]);
    }

    // Free resources
    for (size_t i = 0; i < num_neurons; i++) {
        free(neuron_models[i]);
    }
    free(input);
    spiking_free(&spiking_layer);
}

void network_test() { 
    size_t input_size = 28 * 28 * 2; // Example input size for 2-channel image
    float *input = (float *)malloc(input_size * sizeof(float));

    for (size_t i = 0; i < input_size; i++) {
        input[i] = (float)i / input_size;
    }

    // Create a network with 3 layers
    Network *network = create_network(3);

    // Initialize and add Conv2D layer
    Conv2DLayer *conv_layer = (Conv2DLayer *)malloc(sizeof(Conv2DLayer));
    conv2d_initialize(conv_layer, 2, 12, 5, 1, 0, 24);
    add_layer(network, (LayerBase *)conv_layer, 0);

    // Initialize and add MaxPool2D layer
    MaxPool2DLayer *pool_layer = (MaxPool2DLayer *)malloc(sizeof(MaxPool2DLayer));
    maxpool2d_initialize(pool_layer, 2, 2, 0, 28, 2);
    add_layer(network, (LayerBase *)pool_layer, 1);

    // Initialize and add Flatten layer
    FlattenLayer *flatten_layer = (FlattenLayer *)malloc(sizeof(FlattenLayer));
    flatten_initialize(flatten_layer, conv_layer->output_size);
    add_layer(network, (LayerBase *)flatten_layer, 2);

    // Perform forward pass
    forward(network, input, input_size);

    // Free resources
    free(input);
    free_network(network);
}

void network_loader_test() { 
    // Load the network from the config file
    Network *network = initialize_network_from_file("example_model.json");
    if (!network) {
        printf("Failed to initialize network.\n");
    }

    // Generate synthetic input (28x28 image with 2 channels)
    size_t input_size = 28 * 28 * 2;
    float *input = (float *)malloc(input_size * sizeof(float));
    generate_synthetic_input(input, input_size);

    // Perform forward pass
    for(int i = 0; i < 10; ++i) {
        forward(network, input, input_size);
    }

    // Assume last layer is a spiking layer and print spikes
    SpikingLayer *last_layer = (SpikingLayer *)network->layers[network->num_layers - 1];
    print_output_spikes(last_layer->base.output, last_layer->num_neurons);

    // Free resources
    free(input);
    free_network(network); 
}

void print_sample(const NMNISTSample *sample, size_t max_events_to_display) {
    printf("Label: %d\n", sample->label);
    printf("Number of events: %zu\n", sample->num_events);

    size_t num_events_to_display = sample->num_events < max_events_to_display
                                       ? sample->num_events
                                       : max_events_to_display;
    for (size_t i = 0; i < num_events_to_display; i++) {
        printf("Event %zu: X=%d, Y=%d, Polarity=%d, Timestamp=%u\n", i + 1,
               sample->events[i].x, sample->events[i].y,
               sample->events[i].polarity, sample->events[i].timestamp);
    }

    if (sample->num_events > max_events_to_display) {
        printf("... (%zu more events not displayed)\n",
               sample->num_events - max_events_to_display);
    }
}

void nmnist_loader_test() {
    // Directory containing the NMNIST dataset
    // const char *dataset_dir = "/Users/karol/Desktop/karol/agh/praca-snn/N-MNIST/Train";   
    const char *dataset_dir = "C:/Users/karol/Documents/datasets/N-MNIST/Test";   


    // Maximum number of samples to load for testing
    size_t max_samples_to_load = 10000;

    // Enable stabilization (true) or disable it (false)
    bool stabilize = true;

    printf("Loading NMNIST dataset with stabilization=%s...\n",
           stabilize ? "ENABLED" : "DISABLED");

    // Load the NMNIST dataset
    NMNISTDataset *dataset =
        load_nmnist_dataset(dataset_dir, max_samples_to_load, stabilize);

    if (!dataset) {
        printf("Error: Failed to load NMNIST dataset.\n");
        return;
    }

    printf("Loaded %zu samples from NMNIST dataset.\n", dataset->num_samples);

    // Display information for each loaded sample
    for (size_t i = 0; i < 10; i++) {
        //printf("\nSample %zu:\n", i + 1);
        if(&dataset->samples[i].label != 0) {
            visualize_sample_frames(&dataset->samples[i], "out/sample_01_frames", 16, 28, 28, 100000);
            break;
        }
        //print_sample(&dataset->samples[i], 10); // Display up to 10 events per sample
    }

    // Free the dataset
    free_nmnist_dataset(dataset);

    printf("NMNIST dataset successfully tested and freed.\n");
    return;
}

void discretization_test() {
    // Load the NMNIST dataset
    //NMNISTDataset *dataset = load_nmnist_dataset("/Users/karol/Desktop/karol/agh/praca-snn/N-MNIST/Train", 10, true);

    NMNISTSample sample = load_nmnist_sample(
    "/Users/karol/Desktop/karol/agh/praca-snn/N-MNIST/Train/5/00333.bin",
    4, false);

    int max_time = sample.events[sample.num_events - 1].timestamp;

    //NMNISTSample sample = load_nmnist_sample(file_path, digit, stabilize);
    visualize_sample_frames(&sample, "out/sample_0_frames", 311, 34, 34, max_time);
    plot_event_grid(&sample, 2, 3, 0);
    
    /*
    // Visualize the first sample as temporal frames
    if (dataset->num_samples > 0) {
        visualize_sample_frames(&dataset->samples[1], "out/sample_1_frames", 16, 34, 34, 100000);
        visualize_sample_frames(&dataset->samples[2], "out/sample_2_frames", 16, 34, 34, 100000);
        visualize_sample_frames(&dataset->samples[3], "out/sample_3_frames", 16, 34, 34, 100000);
    }

    // Free the dataset
    free_nmnist_dataset(dataset);
    */
}


void train_test() {
    const char *network_config_path = "C:/Users/karol/Desktop/karol/agh/praca_snn/N-MNIST/Train/Train";   
    const char *dataset_path = "torch_model.json";

    // Load the network
    printf("Loading network from %s...\n", dataset_path);
    Network *network = initialize_network_from_file(dataset_path);
    if (!network) {
        printf("Error: Failed to load network.\n");
        return;
    }

    // Load the NMNIST dataset
    printf("Loading dataset from %s...\n", network_config_path);
    NMNISTDataset *dataset = load_nmnist_dataset(network_config_path, 160, true); // Load up to 160 samples
    if (!dataset) {
        printf("Error: Failed to load dataset.\n");
        free_network(network);
        return;
    }

    // Train the network
    printf("Training the network...\n");
    train(network, dataset);

    // Clean up
    free_nmnist_dataset(dataset);
    free_network(network);

    printf("Training test completed successfully.\n");
}

// Simple synthetic data generator
float* generate_batch(int batch_size, int features, int* labels) {
    float* data = malloc(batch_size * features * sizeof(float));
    for (int i = 0; i < batch_size; i++) {
        float sum = 0;
        for (int j = 0; j < features; j++) {
            data[i * features + j] = (float)rand() / RAND_MAX;
            sum += data[i * features + j];
        }
        labels[i] = (sum / features) > 0.5f ? 1 : 0;
    }
    return data;
}

// Layer creation functions implementation
Conv2DLayer* create_conv2d_layer(int height, int width, int in_channels, int out_channels, 
                               int kernel_size, int stride, int padding) {
    Conv2DLayer *layer = (Conv2DLayer*)malloc(sizeof(Conv2DLayer));
    conv2d_initialize(layer, in_channels, out_channels, kernel_size, stride, padding, height);
    return layer;
}

MaxPool2DLayer* create_maxpool_layer(int height, int width, int channels, 
                                   int pool_size, int stride) {
    MaxPool2DLayer *layer = (MaxPool2DLayer*)malloc(sizeof(MaxPool2DLayer));
    maxpool2d_initialize(layer, pool_size, stride, 0, height, channels);
    return layer;
}

FlattenLayer* create_flatten_layer(size_t input_size) {
    FlattenLayer *layer = (FlattenLayer*)malloc(sizeof(FlattenLayer));
    flatten_initialize(layer, input_size);
    return layer;
}

LinearLayer* create_linear_layer(size_t in_features, size_t out_features) {
    LinearLayer *layer = (LinearLayer*)malloc(sizeof(LinearLayer));
    linear_initialize(layer, in_features, out_features);
    return layer;
}

SpikingLayer* create_spiking_layer(size_t num_neurons) {
    SpikingLayer *layer = (SpikingLayer*)malloc(sizeof(SpikingLayer));
    // Using LIF neurons as default
    ModelBase **neurons = (ModelBase**)malloc(num_neurons * sizeof(ModelBase*));
    for (size_t i = 0; i < num_neurons; i++) {
        neurons[i] = (ModelBase*)malloc(sizeof(LIFNeuron));
        // Initialize LIF neuron here

        lif_initialize((LIFNeuron *)neurons[i], 0.0f, 0.5f, 0.0f, 0.5f);
    }
    spiking_initialize(layer, num_neurons, neurons);
    return layer;
}

// Iris dataset parameters
#define MAX_LINE_LENGTH 100
#define NUM_FEATURES 4
#define NUM_CLASSES 3
#define NUM_SAMPLES 150
#define TRAIN_SIZE 120  // 80% of 150
#define TEST_SIZE 30    // 20% of 150

typedef struct {
    float sepal_length;
    float sepal_width;
    float petal_length;
    float petal_width;
    int variety;  // 0=Setosa, 1=Versicolor, 2=Virginica
} IrisSample;

void parse_iris_line(char *line, IrisSample *sample) {
    char *token;
    int col = 0;
    
    // Remove newline if present
    line[strcspn(line, "\n")] = 0;
    
    token = strtok(line, ",");
    while (token != NULL && col < 5) {
        // Remove quotes if present
        if (token[0] == '"') {
            memmove(token, token+1, strlen(token));  // Remove opening quote
            token[strlen(token)-1] = '\0';           // Remove closing quote
        }
        
        switch(col) {
            case 0:  // sepal.length
                sample->sepal_length = atof(token);
                break;
            case 1:  // sepal.width
                sample->sepal_width = atof(token);
                break;
            case 2:  // petal.length
                sample->petal_length = atof(token);
                break;
            case 3:  // petal.width
                sample->petal_width = atof(token);
                break;
            case 4:  // variety
                if (strcmp(token, "\"Setosa\"") == 0 || strcmp(token, "Setosa") == 0)
                    sample->variety = 0;
                else if (strcmp(token, "\"Versicolor\"") == 0 || strcmp(token, "Versicolor") == 0)
                    sample->variety = 1;
                else if (strcmp(token, "\"Virginica\"") == 0 || strcmp(token, "Virginica") == 0)
                    sample->variety = 2;
                break;
        }
        
        token = strtok(NULL, ",");
        col++;
    }
}

void calculate_mean_std(IrisSample *samples, int count, float *means, float *stds) {
    // Initialize sums and squares
    float sums[NUM_FEATURES] = {0};
    float squares[NUM_FEATURES] = {0};
    
    // Calculate sums
    for (int i = 0; i < count; i++) {
        sums[0] += samples[i].sepal_length;
        sums[1] += samples[i].sepal_width;
        sums[2] += samples[i].petal_length;
        sums[3] += samples[i].petal_width;
        
        squares[0] += samples[i].sepal_length * samples[i].sepal_length;
        squares[1] += samples[i].sepal_width * samples[i].sepal_width;
        squares[2] += samples[i].petal_length * samples[i].petal_length;
        squares[3] += samples[i].petal_width * samples[i].petal_width;
    }
    
    // Calculate means and standard deviations
    for (int i = 0; i < NUM_FEATURES; i++) {
        means[i] = sums[i] / count;
        stds[i] = sqrt((squares[i] / count) - (means[i] * means[i]));
    }
}

void standardize_features(IrisSample *samples, int count, float *means, float *stds) {
    for (int i = 0; i < count; i++) {
        samples[i].sepal_length = (samples[i].sepal_length - means[0]) / stds[0];
        samples[i].sepal_width = (samples[i].sepal_width - means[1]) / stds[1];
        samples[i].petal_length = (samples[i].petal_length - means[2]) / stds[2];
        samples[i].petal_width = (samples[i].petal_width - means[3]) / stds[3];
    }
}

void load_iris_dataset(const char *filename, 
                      float X_train[TRAIN_SIZE][NUM_FEATURES], int y_train[TRAIN_SIZE],
                      float X_test[TEST_SIZE][NUM_FEATURES], int y_test[TEST_SIZE]) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    
    IrisSample samples[NUM_SAMPLES];
    char line[MAX_LINE_LENGTH];
    int index = 0;
    
    // Skip header line
    fgets(line, sizeof(line), file);
    
    // Read all samples
    while (fgets(line, sizeof(line), file) != NULL && index < NUM_SAMPLES) {
        parse_iris_line(line, &samples[index]);
        index++;
    }
    fclose(file);
    
    // Calculate mean and std for standardization
    float means[NUM_FEATURES], stds[NUM_FEATURES];
    calculate_mean_std(samples, NUM_SAMPLES, means, stds);
    
    // Standardize features (like StandardScaler)
    standardize_features(samples, NUM_SAMPLES, means, stds);
    
    // Shuffle samples (for proper train/test split)
    for (int i = NUM_SAMPLES - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        IrisSample temp = samples[i];
        samples[i] = samples[j];
        samples[j] = temp;
    }
    
    // Split into train and test sets (80/20)
    for (int i = 0; i < TRAIN_SIZE; i++) {
        X_train[i][0] = samples[i].sepal_length;
        X_train[i][1] = samples[i].sepal_width;
        X_train[i][2] = samples[i].petal_length;
        X_train[i][3] = samples[i].petal_width;
        y_train[i] = samples[i].variety;
    }
    
    for (int i = 0; i < TEST_SIZE; i++) {
        X_test[i][0] = samples[TRAIN_SIZE + i].sepal_length;
        X_test[i][1] = samples[TRAIN_SIZE + i].sepal_width;
        X_test[i][2] = samples[TRAIN_SIZE + i].petal_length;
        X_test[i][3] = samples[TRAIN_SIZE + i].petal_width;
        y_test[i] = samples[TRAIN_SIZE + i].variety;
    }
}

// ReLU activation function
void relu_forward(float *input, size_t size) {
    for (size_t i = 0; i < size; i++) {
        input[i] = input[i] > 0 ? input[i] : 0;
 
 
    }
}

// Softmax function
void softmax(float *input, size_t size) {
    float max_val = input[0];
    for (size_t i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        input[i] = expf(input[i] - max_val);
        sum += input[i];
    }
    
    for (size_t i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

// Cross-entropy loss
float cross_entropy_loss(float* predictions, int target, size_t num_classes) {
    float loss = 0.0f;
    for (size_t i = 0; i < num_classes; i++) {
        if (i == target) {
            loss += -logf(predictions[i] + 1e-10f); // Small epsilon to avoid log(0)
        }
    }
    return loss;
}

// Calculate accuracy
float calculate_accuracy(int *predicted, int *true_labels, size_t size) {
    int correct = 0;
    for (size_t i = 0; i < size; i++) {
        if (predicted[i] == true_labels[i]) {
            correct++;
        }
    }
    return (float)correct / size;
}

// helper to shuffle the dataset
void shuffle_data(float X[][NUM_FEATURES], int y[], int size, int num_features) {
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        
        // Swap features
        float temp_features[num_features];
        memcpy(temp_features, X[i], num_features * sizeof(float));
        memcpy(X[i], X[j], num_features * sizeof(float));
        memcpy(X[j], temp_features, num_features * sizeof(float));
        
        // Swap labels
        int temp_label = y[i];
        y[i] = y[j];
        y[j] = temp_label;
    }
}

void scale_gradients_by_batch(Network *network, int batch_size) {
    for (int l = 0; l < network->num_layers; l++) {
        LayerBase *layer = network->layers[l];

        // Cast the layer based on its type
        if (layer->layer_type == LAYER_CONV2D) {
            Conv2DLayer *conv_layer = (Conv2DLayer*)layer;
            for (int i = 0; i < conv_layer->base.num_weights; i++) {
                conv_layer->base.weight_gradients[i] /= (float)batch_size;
            }
            for (int i = 0; i < conv_layer->out_channels; i++) {
                conv_layer->base.bias_gradients[i] /= (float)batch_size;
            }
        } else if (layer->layer_type == LAYER_LINEAR) {
            LinearLayer *linear_layer = (LinearLayer*)layer;
            for (int i = 0; i < linear_layer->base.num_weights; i++) {
                linear_layer->base.weight_gradients[i] /= (float)batch_size;
            }
            for (int i = 0; i < linear_layer->out_features; i++) {
                linear_layer->base.bias_gradients[i] /= (float)batch_size;
            }
        } 
    }
}

void save_hidden_gradients_to_file(const float* hidden_gradients, int batch_num, int sample_num, int size, const char* filename) {
    FILE* file = fopen(filename, "a"); // Append mode
    if (file == NULL) {
        perror("Failed to open file");
        return;
    }

    fprintf(file, "Batch %d, Sample %d:\n", batch_num, sample_num);
    for (int i = 0; i < size; i++) {
        fprintf(file, "%f ", hidden_gradients[i]);
    }
    fprintf(file, "\n\n");

    fclose(file);
}

void save_output_to_file(const float* output, int batch_num, int sample_num, const char* filename) {
    FILE* file = fopen(filename, "a"); // Append mode
    if (file == NULL) {
        perror("Failed to open file");
        return;
    }

    fprintf(file, "Batch %d, Sample %d:\n", batch_num, sample_num);
    for (int i = 0; i < NUM_CLASSES; i++) {
        fprintf(file, "%f ", output[i]);
    }
    fprintf(file, "\n\n");

    fclose(file);
}

void iris_classification_example() {
    srand(time(NULL));  // For reproducibility
    
    // Load dataset
    float X_train[TRAIN_SIZE][NUM_FEATURES];
    int y_train[TRAIN_SIZE];
    float X_test[TEST_SIZE][NUM_FEATURES];
    int y_test[TEST_SIZE];
    
    load_iris_dataset("C:/Users/karol/Desktop/karol/agh/praca_snn/iris.csv", X_train, y_train, X_test, y_test);
    
    // Initialize network
    Network network;
    network.num_layers = 2;  // Linear(4,16) -> ReLU -> Linear(16,3)
    network.layers = malloc(network.num_layers * sizeof(LayerBase*));
    
    // Create layers
    add_layer(&network, (LayerBase*)create_linear_layer(NUM_FEATURES, 16), 0);
    add_layer(&network, (LayerBase*)create_linear_layer(16, NUM_CLASSES), 1);
    
    // Training parameters
    const int epochs = 100;
    const float learning_rate = 0.02f;
    
    
    const int batch_size = 16;
    printf("Starting training with batch size %d...\n", batch_size);

    // Buffers for batch processing
    float batch_X[batch_size][NUM_FEATURES];
    int batch_y[batch_size];
    float batch_hidden_activations[batch_size][16];
    float batch_output_activations[batch_size][NUM_CLASSES];

    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;

        shuffle_data(X_train, y_train, TRAIN_SIZE, NUM_FEATURES);

        for (int batch_start = 0; batch_start < TRAIN_SIZE; batch_start += batch_size) {
            int current_batch_size = (batch_start + batch_size <= TRAIN_SIZE) ? batch_size : TRAIN_SIZE - batch_start;

            float *grads = ((LinearLayer *)network.layers[0])->base.weight_gradients;
            float batch_loss = 0.0f;


            zero_grads(&network);  

            // Prepare batch data
            for (int i = 0; i < current_batch_size; i++) {
                memcpy(batch_X[i], X_train[batch_start + i], NUM_FEATURES * sizeof(float));
                batch_y[i] = y_train[batch_start + i];
            }

            // Forward pass over the batch
            for (int i = 0; i < current_batch_size; i++) {
                network.layers[0]->forward(network.layers[0], batch_X[i], NUM_FEATURES);
                //memcpy(batch_hidden_activations[i], network.layers[0]->output, 16 * sizeof(float));
                //relu_forward(batch_hidden_activations[i], 16);  // In-place ReLU
                relu_forward(network.layers[0]->output, 16);  // apply directly to .output
                memcpy(batch_hidden_activations[i], network.layers[0]->output, 16 * sizeof(float)); // optional, for reuse

                network.layers[1]->forward(network.layers[1], batch_hidden_activations[i], 16);
                memcpy(batch_output_activations[i], network.layers[1]->output, NUM_CLASSES * sizeof(float));
                softmax(batch_output_activations[i], NUM_CLASSES);

                // save_output_to_file(batch_output_activations[i], batch_start, i, "out/iris_outputs_batch.txt");
                // epoch_loss += cross_entropy_loss(batch_output_activations[i], batch_y[i], NUM_CLASSES);

                batch_loss += cross_entropy_loss(batch_output_activations[i], batch_y[i], NUM_CLASSES);
            }

            epoch_loss += batch_loss;
            

            // Backward pass for each sample
            for (int i = 0; i < current_batch_size; i++) {
                float output_gradients[NUM_CLASSES];
                for (int c = 0; c < NUM_CLASSES; c++) {
                    // Do NOT divide by batch size here!
                    output_gradients[c] = batch_output_activations[i][c] - (c == batch_y[i] ? 1.0f : 0.0f);
                }

                float* hidden_gradients = network.layers[1]->backward(network.layers[1], output_gradients);

                // Apply ReLU derivative
                for (int h = 0; h < 16; h++) {
                    hidden_gradients[h] *= (batch_hidden_activations[i][h] > 0) ? 1.0f : 0.0f;
                }

              //  save_hidden_gradients_to_file(hidden_gradients, batch_start / batch_size, i, 16, "out/hidden_gradients_log.txt");


                network.layers[0]->backward(network.layers[0], hidden_gradients);
            }

            float *gradsW = ((LinearLayer *)network.layers[0])->base.weight_gradients;

            // Now divide accumulated gradients by batch size
            //scale_gradients_by_batch(&network, current_batch_size);

            // Update weights
            update_weights(&network, learning_rate);
            
        }

        log_gradients(&network, epoch, 0);
    
        if ((epoch + 1) % 10 == 0) {
            printf("Epoch [%d/%d], Loss: %.4f\n", epoch + 1, epochs, epoch_loss / TRAIN_SIZE);
        }
    }
    
    // Evaluation (same as before)
    int predictions[TEST_SIZE];
    float test_loss = 0.0f;
    float hidden_activations[16];
    float output_activations[NUM_CLASSES];
    
    for (int i = 0; i < TEST_SIZE; i++) {
        // Forward pass (no need for gradients during evaluation)
        network.layers[0]->forward(network.layers[0], X_test[i], NUM_FEATURES);
        memcpy(hidden_activations, network.layers[0]->output, sizeof(float) * 16);
        relu_forward(hidden_activations, 16);
        network.layers[1]->forward(network.layers[1], hidden_activations, 16);
        memcpy(output_activations, network.layers[1]->output, sizeof(float) * NUM_CLASSES);
        softmax(output_activations, NUM_CLASSES);
        
        // Get predicted class
        int pred_class = 0;
        float max_prob = output_activations[0];
        for (int c = 1; c < NUM_CLASSES; c++) {
            if (output_activations[c] > max_prob) {
                max_prob = output_activations[c];
                pred_class = c;
            }
        }
        predictions[i] = pred_class;
        
        // Calculate loss
        test_loss += cross_entropy_loss(output_activations, y_test[i], NUM_CLASSES);
    }
    
    // Calculate accuracy
    float accuracy = calculate_accuracy(predictions, y_test, TEST_SIZE);
    printf("\nTest Accuracy: %.2f%%\n", accuracy * 100);
    
    // Cleanup
    for (size_t i = 0; i < network.num_layers; i++) {
        free(network.layers[i]);
    }
    free(network.layers);
}

// Create the simple network
Network* create_simple_net() {
    Network* net = create_network(4); // conv, pool, flatten, linear
    
    // Create layers
    Conv2DLayer* conv = malloc(sizeof(Conv2DLayer));
    conv2d_initialize(conv, 1, 2, 3, 1, 0, 6); // in_channels, out_channels, kernel_size, stride, padding, input_dim
    
    MaxPool2DLayer* pool = malloc(sizeof(MaxPool2DLayer));
    maxpool2d_initialize(pool, 2, 2, 0, 4, 2); // kernel_size, stride, padding, input_dim, num_channels
    
    FlattenLayer* flatten = malloc(sizeof(FlattenLayer));
    flatten_initialize(flatten, 2 * 2 * 2); // output size 2*2*2
    
    LinearLayer* linear = malloc(sizeof(LinearLayer));
    linear_initialize(linear, 2 * 2 * 2, 3); // in_features, out_features
    
    // Add layers to network
    add_layer(net, (LayerBase*)conv, 0);
    add_layer(net, (LayerBase*)pool, 1);
    add_layer(net, (LayerBase*)flatten, 2);
    add_layer(net, (LayerBase*)linear, 3);
    
    return net;
}

// Create the synthetic dataset
void create_dataset(float*** images, int** labels, int* num_samples) {
    *num_samples = 320; // 10 each of 3 patterns
    *images = malloc(*num_samples * sizeof(float*));
    *labels = malloc(*num_samples * sizeof(int));
    
    // Horizontal line pattern (label 0)
    float horizontal[6][6] = {
        {0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1, 1},
        {0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0}
    };
    
    // Vertical line pattern (label 1)
    float vertical[6][6] = {
        {0, 0, 1, 0, 0, 0},
        {0, 0, 1, 0, 0, 0},
        {0, 0, 1, 0, 0, 0},
        {0, 0, 1, 0, 0, 0},
        {0, 0, 1, 0, 0, 0},
        {0, 0, 1, 0, 0, 0}
    };
    
    // Diagonal line pattern (label 2)
    float diagonal[6][6] = {
        {1, 0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0},
        {0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 1}
    };
    
    // Create 10 copies of each pattern
    for (int i = 0; i < *num_samples ; i++) {
        // Horizontal
        (*images)[i] = malloc(6 * 6 * sizeof(float));
        memcpy((*images)[i], horizontal, sizeof(horizontal));
        (*labels)[i] = 0;
        
        // Vertical
        (*images)[i+*num_samples] = malloc(6 * 6 * sizeof(float));
        memcpy((*images)[i+*num_samples], vertical, sizeof(vertical));
        (*labels)[i+*num_samples] = 1;
        
        // Diagonal
        (*images)[i+*num_samples*2] = malloc(6 * 6 * sizeof(float));
        memcpy((*images)[i+*num_samples*2], diagonal, sizeof(diagonal));
        (*labels)[i+*num_samples*2] = 2;
    }
}

// Free the dataset
void free_dataset(float** images, int* labels, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        free(images[i]);
    }
    free(images);
    free(labels);
}

// Training function
void test_hor_vert_dataset(Network* net, float** images, int* labels, int num_samples, int epochs, float learning_rate) {
    int batch_size = 8;
    int num_batches = num_samples / batch_size;
    float* gradients = malloc(3 * sizeof(float));

    // Temporary storage for batch outputs and gradients
    float** batch_outputs = (float**)malloc(batch_size * sizeof(float*));
    int* batch_labels = (int*)malloc(batch_size * sizeof(int));
    float** batch_gradients = (float**)malloc(batch_size * sizeof(float*));
    float* avg_gradients = NULL;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        
        // Training loop - more PyTorch-like batching
    for (int batch = 0; batch < num_batches; batch++) {
        zero_grads(net);  // Reset all gradients at start of batch
        
        float batch_loss = 0.0f;
        int actual_batch_size = 0;
        
        // Temporary storage for batch outputs and gradients
        float** batch_outputs = (float**)malloc(batch_size * sizeof(float*));
        int* batch_labels = (int*)malloc(batch_size * sizeof(int));
        float** batch_gradients = (float**)malloc(batch_size * sizeof(float*));
        
        // Forward pass for entire batch
        for (int i = 0; i < batch_size; i++) {
            int idx = batch * batch_size + i;
            if (idx >= num_samples) break;
            
            actual_batch_size++;
            
            batch_labels[i] = labels[idx];
            forward(net, images[idx], 6 * 6);
            
            // Get output and apply softmax
            float* output = net->layers[net->num_layers - 1]->output;
            softmax(output, 3);
            
            // Store outputs
            batch_outputs[i] = (float*)malloc(3 * sizeof(float));
            memcpy(batch_outputs[i], output, 3 * sizeof(float));
            
            // Calculate loss
            batch_loss += cross_entropy_loss(output, labels[idx], 3);
        }
        
        // Backward pass for entire batch
        for (int i = 0; i < actual_batch_size; i++) {
            float* gradients = (float*)malloc(3 * sizeof(float));
            for (int j = 0; j < 3; j++) {
                gradients[j] = batch_outputs[i][j] - (batch_labels[i] == j ? 1.0f : 0.0f);
            }
            
            // Store gradients for accumulation
            batch_gradients[i] = gradients;
        }
        
        // Average gradients across batch
        float* avg_gradients = (float*)calloc(3, sizeof(float));
        for (int i = 0; i < actual_batch_size; i++) {
            for (int j = 0; j < 3; j++) {
                avg_gradients[j] += batch_gradients[i][j];
            }
        }
        for (int j = 0; j < 3; j++) {
            avg_gradients[j] /= actual_batch_size;
        }
        
        // Backpropagate averaged gradients
        float* current_gradients = avg_gradients;
        for (int l = net->num_layers - 1; l >= 0; l--) {
            current_gradients = net->layers[l]->backward(net->layers[l], current_gradients);
        }
        
        // Update weights
        update_weights(net, learning_rate);
        
        // Free temporary memory
        for (int i = 0; i < actual_batch_size; i++) {
            free(batch_outputs[i]);
            free(batch_gradients[i]);
        }
        free(batch_outputs);
        free(batch_labels);
        free(batch_gradients);
        free(avg_gradients);
        
        total_loss += batch_loss / actual_batch_size;
        }

        log_gradients(net, epoch, 0);
        printf("Epoch %d, Loss: %.4f\n", epoch + 1, total_loss / num_batches);
    }
}

// Test function
void test(Network* net, float** images, int* labels, int num_samples) {
    int correct = 0;
    
    for (int i = 0; i < num_samples; i++) {
        forward(net, images[i], 6 * 6);
        
        // Get output and apply softmax
        float* output = net->layers[net->num_layers - 1]->output;
        softmax(output, 3);
        
        // Get predicted class
        int predicted = 0;
        float max_prob = output[0];
        for (int j = 1; j < 3; j++) {
            if (output[j] > max_prob) {
                max_prob = output[j];
                predicted = j;
            }
        }
        
        if (predicted == labels[i]) {
            correct++;
        }
    }
    
    printf("Accuracy: %.2f%%\n", (float)correct / num_samples * 100.0f);
}

void hor_vert_dataset() {
    // Create dataset
    float** images;
    int* labels;
    int num_samples;
    create_dataset(&images, &labels, &num_samples);
    
    // Create network
    Network* net = create_simple_net();
    
    // Train network
    test_hor_vert_dataset(net, images, labels, num_samples, 10, 0.01f);
    
    // Test network
    test(net, images, labels, num_samples);
    
    // Cleanup
    free_network(net);
    free_dataset(images, labels, num_samples);
}



