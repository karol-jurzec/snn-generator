#ifndef SPIKING_LAYER_H
#define SPIKING_LAYER_H

#include <stddef.h>
#include "../models/lif_neuron.h"
#include "layer_base.h"
#include <string.h>
#include <json-c/json.h>

typedef struct {
    LayerBase base;             // Inherits LayerBase
    LIFLayer neuron_layer;      // Now uses SoA structure
    float *output_spikes;       // Output spikes (1 if spiked, 0 otherwise)
    float *input_gradients;     // Gradient of input to propagate backward
    float *spike_gradients;     // Surrogate gradients for spikes
    float *membrane_history;    // [time_steps][num_neurons]
    int *spike_history;         // [time_steps][num_neurons]
} SpikingLayer;

void spiking_initialize(SpikingLayer *layer, size_t num_neurons, float v_rest, float threshold, float v_reset, float beta);
void spiking_forward(void *self, float *input, size_t input_size, size_t time_step);
float* spiking_backward(void *self, float *gradients, size_t time_step);
void spiking_update_weights(void *self, float learning_rate);
void spiking_zero_grad(void *self);
void spiking_free(SpikingLayer *layer);
void spiking_reset_spike_counts(void *self);

#endif // SPIKING_LAYER_H