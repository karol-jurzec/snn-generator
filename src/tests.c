#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/tests.h"
#include "../include/network.h"

#include "../include/models/model_base.h"
#include "../include/models/lif_neuron.h"

#include "../include/utils/snn_plot.h"
#include "../include/utils/layer_utils.h"
#include "../include/utils/network_loader.h"
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
    conv2d_initialize(&conv_layer, in_channels, out_channels, kernel_size, stride, padding);

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
    maxpool2d_initialize(&pool_layer, kernel_size, stride, padding);
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
    conv2d_initialize(conv_layer, 2, 12, 5, 1, 0);
    add_layer(network, (LayerBase *)conv_layer, 0);

    // Initialize and add MaxPool2D layer
    MaxPool2DLayer *pool_layer = (MaxPool2DLayer *)malloc(sizeof(MaxPool2DLayer));
    maxpool2d_initialize(pool_layer, 2, 2, 0);
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
    const char *dataset_dir = "/Users/karol/Desktop/karol/agh/praca-snn/N-MNIST/Train";   

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
        printf("\nSample %zu:\n", i + 1);
        print_sample(&dataset->samples[i], 10); // Display up to 10 events per sample
    }

    // Free the dataset
    free_nmnist_dataset(dataset);

    printf("NMNIST dataset successfully tested and freed.\n");
    return;
}

