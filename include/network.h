#ifndef NETWORK_H
#define NETWORK_H

#include <stddef.h>
#include "layers/layer_base.h"
#include "utils/nmnist_loader.h"

#define NETWORK_FORWARD_FUNC(name) void (*name)(void *self, float *input, size_t input_size)

typedef struct {
    NETWORK_FORWARD_FUNC(forward);  // Forward function pointer for layers


    LayerBase **layers;   // Array of pointers to layers (polymorphic)
    size_t num_layers;    // Number of layers in the network
} Network;

Network *create_network(size_t num_layers);
void add_layer(Network *network, LayerBase *layer, size_t index);
void forward(Network *network, float *input, size_t input_size, int time_step);
void free_network(Network *network);
float calculate_loss(float *output, int label, size_t output_size);
void update_weights(Network *network, float learning_rate);
void train(Network *network, NMNISTDataset *dataset);
float test(Network *network, NMNISTDataset *dataset);
void zero_grads(Network* model);
void compute_probabilities(float *spike_counts, size_t num_neurons, float *probabilities);




#endif // NETWORK_H
