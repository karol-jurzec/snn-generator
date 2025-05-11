#ifndef SPIKING_LAYER_H
#define SPIKING_LAYER_H

#include <stddef.h>
#include "../models/model_base.h"  // Base model for neurons
#include "layer_base.h"            // Base layer  

typedef struct {
    LayerBase base;             // Inherits LayerBase for polymorphic forward
    ModelBase **neurons;        // Array of pointers to neurons (polymorphic)
    size_t num_neurons;         // Number of neurons in the layer
    float *input; 
    float *output_spikes;       // Output spikes (1 if spiked, 0 otherwise)

    // Training-specific
    float *input_gradients;     // Gradient of input to propagate backward
    float *spike_gradients;     // Surrogate gradients for spikes

    float *membrane_history;  // [time_steps][num_neurons]
    int *spike_history;       // [time_steps][num_neurons]
} SpikingLayer;

void spiking_initialize(SpikingLayer *layer, size_t num_neurons, ModelBase **neuron_models);
void spiking_forward(void *self, float *input, size_t input_size, size_t time_step);
float* spiking_backward(void *self, float *gradients, size_t time_step);
void spiking_update_weights(void *self, float learning_rate); // If needed for trainable weights
void spiking_zero_grad(void *self);
void spiking_free(SpikingLayer *layer);
void spiking_reset_spike_counts(void *self);

#endif // SPIKING_LAYER_H
