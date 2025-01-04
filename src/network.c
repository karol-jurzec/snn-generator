#include <stdlib.h>
#include <stdio.h>

#include "../include/network.h"

// Create a new network with a given number of layers
Network *create_network(size_t num_layers) {
    Network *network = (Network *)malloc(sizeof(Network));
    network->layers = (LayerBase **)malloc(num_layers * sizeof(LayerBase *));
    network->num_layers = num_layers;
    return network;
}

// Add a layer to the network at a specific index
void add_layer(Network *network, LayerBase *layer, size_t index) {
    if (index < network->num_layers) {
        network->layers[index] = layer;
    } else {
        printf("Error: Index out of bounds when adding layer.\n");
    }
}

// Perform forward propagation through the network
void forward(Network *network, float *input, size_t input_size) {
    float *current_input = input;
    size_t current_input_size = input_size;

    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        layer->forward(layer, current_input, current_input_size);

        // Update current input to the output of the current layer
        current_input = layer->output; // Correctly reference the layer's output
        current_input_size = layer->output_size; // Update input size for the next layer
        
        for (size_t i = 0; i < layer->output_size && i < 10; i++) {
            printf("output[%zu] = %f\n", i, layer->output[i]);
        }
    }
}

// Free the allocated memory for the network
void free_network(Network *network) {
    for (size_t i = 0; i < network->num_layers; i++) {
        free(network->layers[i]);
    }
    free(network->layers);
    free(network);
}
