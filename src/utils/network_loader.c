#include "../../include/utils/network_loader.h"

#include "../../include/layers/conv2d_layer.h"
#include "../../include/layers/maxpool2d_layer.h"
#include "../../include/layers/flatten_layer.h"
#include "../../include/layers/linear_layer.h"
#include "../../include/layers/spiking_layer.h"

#include "../../include/models/lif_neuron.h"

#include <json-c/json.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

ModelBase **initialize_neurons(const char *neuron_type, size_t num_neurons, struct json_object *neuron_config) {
    ModelBase **neurons = (ModelBase **)malloc(num_neurons * sizeof(ModelBase *));
    
    if (strcmp(neuron_type, "Leaky") == 0) {
        float beta = json_object_get_double(json_object_object_get(neuron_config, "beta"));
        for (size_t i = 0; i < num_neurons; i++) {
            neurons[i] = (ModelBase *)malloc(sizeof(LIFNeuron));
            lif_initialize((LIFNeuron *)neurons[i], 0.0f, 1.0f, 0.0f, beta);
        }
    }
    // rest neuron types etc... 

    return neurons;
}

void parse_and_add_layer(Network *network, struct json_object *layer_config, size_t index) {
    static int current_channels = 2;   // Input channels (e.g., for NMNIST: 2 polarity channels)
    static int current_height = 28;   // Input height
    static int current_width = 28;    // Input width

    const char *type = json_object_get_string(json_object_object_get(layer_config, "type"));

    if (strcmp(type, "Conv2d") == 0) {
        int in_channels = json_object_get_int(json_object_object_get(layer_config, "in_channels"));
        int out_channels = json_object_get_int(json_object_object_get(layer_config, "out_channels"));
        int kernel_size = json_object_get_int(json_object_array_get_idx(json_object_object_get(layer_config, "kernel_size"), 0));
        Conv2DLayer *conv_layer = (Conv2DLayer *)malloc(sizeof(Conv2DLayer));
        conv2d_initialize(conv_layer, in_channels, out_channels, kernel_size, 1, 0);

        add_layer(network, (LayerBase *)conv_layer, index);

        // Update shape: Channels change, spatial dims depend on kernel size and stride
        current_channels = out_channels;
        current_height = (current_height - kernel_size + 2 * 0) / 1 + 1; // Update height
        current_width = (current_width - kernel_size + 2 * 0) / 1 + 1;  // Update width

    } else if (strcmp(type, "MaxPool2d") == 0) {
        int kernel_size = json_object_get_int(json_object_object_get(layer_config, "kernel_size"));
        MaxPool2DLayer *pool_layer = (MaxPool2DLayer *)malloc(sizeof(MaxPool2DLayer));
        maxpool2d_initialize(pool_layer, kernel_size, kernel_size, 0);

        add_layer(network, (LayerBase *)pool_layer, index);

        // Update shape: Spatial dims change due to pooling
        current_height = (current_height - kernel_size) / kernel_size + 1;
        current_width = (current_width - kernel_size) / kernel_size + 1;

    } else if (strcmp(type, "Flatten") == 0) {
        FlattenLayer *flatten_layer = (FlattenLayer *)malloc(sizeof(FlattenLayer));
        int flatten_size = current_channels * current_height * current_width; // Compute size
        flatten_initialize(flatten_layer, flatten_size);

        add_layer(network, (LayerBase *)flatten_layer, index);

    } else if (strcmp(type, "Linear") == 0) {
        int in_features = json_object_get_int(json_object_object_get(layer_config, "in_features"));
        int out_features = json_object_get_int(json_object_object_get(layer_config, "out_features"));
        LinearLayer *linear_layer = (LinearLayer *)malloc(sizeof(LinearLayer));
        linear_initialize(linear_layer, in_features, out_features);
        add_layer(network, (LayerBase *)linear_layer, index);

        // Update shape: Fully connected layers don't affect spatial dims, only output size
        current_channels = out_features; // Next layer will use this as "channels"

    } else if (strcmp(type, "SpikingLayer") == 0) {
        size_t num_neurons = json_object_get_int(json_object_object_get(layer_config, "num_neurons"));
        const char *neuron_type = json_object_get_string(json_object_object_get(layer_config, "neuron_type"));
        ModelBase **neurons = initialize_neurons(neuron_type, num_neurons, layer_config);

        SpikingLayer *spiking_layer = (SpikingLayer *)malloc(sizeof(SpikingLayer));
        spiking_initialize(spiking_layer, num_neurons, neurons);

        add_layer(network, (LayerBase *)spiking_layer, index);

        // Update shape: Spiking layer output size
        current_channels = num_neurons; // Treat as 1D neurons
        current_height = 1;            // No spatial dimensions
        current_width = 1;             // No spatial dimensions
    }
}


Network *initialize_network_from_file(const char *config_path) {
    FILE *fp = fopen(config_path, "r");
    if (!fp) {
        printf("Error opening config file.\n");
        return NULL;
    }

    char buffer[8192];
    fread(buffer, 1, sizeof(buffer), fp);
    fclose(fp);

    struct json_object *parsed_json = json_tokener_parse(buffer);
    struct json_object *layers = NULL;
    json_object_object_get_ex(parsed_json, "layers", &layers);
    size_t num_layers = json_object_array_length(layers);

    Network *network = create_network(num_layers);

    for (size_t i = 0; i < num_layers; i++) {
        struct json_object *layer_config = json_object_array_get_idx(layers, i);
        parse_and_add_layer(network, layer_config, i);
    }

    json_object_put(parsed_json); // Free JSON memory
    return network;
}
