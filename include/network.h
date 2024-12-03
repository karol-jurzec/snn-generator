#ifndef NETWORK_H
#define NETWORK_H

#include <stddef.h>
#include "layers/layer_base.h"

// Structure for the Network
typedef struct {
    LayerBase **layers;   // Array of pointers to layers (polymorphic)
    size_t num_layers;    // Number of layers in the network
} Network;

// Function declarations
Network *create_network(size_t num_layers);
void add_layer(Network *network, LayerBase *layer, size_t index);
void forward(Network *network, float *input, size_t input_size);
void free_network(Network *network);

#endif // NETWORK_H
