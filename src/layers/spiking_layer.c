#include <stdlib.h>
#include <stdio.h>

#include "../../include/layers/spiking_layer.h"
#include "../../include/models/lif_neuron.h"

void spiking_initialize(SpikingLayer *layer, size_t num_neurons, ModelBase **neuron_models) {
    layer->base.layer_type = LAYER_SPIKING;
    layer->base.is_spiking = true;
    layer->base.forward = spiking_forward;
    layer->base.backward = spiking_backward;
    layer->base.reset_spike_counts = spiking_reset_spike_counts;
    layer->base.num_inputs= num_neurons;
    layer->num_neurons = num_neurons;
    layer->neurons = (ModelBase **)malloc(num_neurons * sizeof(ModelBase *));
    layer->base.inputs = (float *)malloc(num_neurons * sizeof(float));
    layer->base.output = (float *)malloc(num_neurons * sizeof(float));
    layer->base.output_size = num_neurons;

    for (size_t i = 0; i < num_neurons; i++) {
        layer->neurons[i] = neuron_models[i];
    }

    layer->spike_gradients = (float *)malloc(num_neurons * sizeof(float));
    layer->base.input_gradients = (float *)malloc(num_neurons * sizeof(float));
}

void spiking_forward(void *self, float *input, size_t input_size, size_t time_step) {
    SpikingLayer *layer = (SpikingLayer *)self;
    
    // Store input for this time step
    memcpy(layer->base.inputs, input, input_size * sizeof(float));

    for (size_t i = 0; i < layer->num_neurons; i++) {
        layer->neurons[i]->update_neuron(layer->neurons[i], input[i]);

        // Store output
        layer->base.output[i] = layer->neurons[i]->spiked;
        
        // Store membrane potential and spikes for BPTT
        if (layer->membrane_history) {
           layer->membrane_history[time_step * layer->num_neurons + i] = layer->neurons[i]->v;
           layer->spike_history[time_step * layer->num_neurons + i] = layer->neurons[i]->spiked;
        }
    }
}

// Backward pass for Spiking Layer
float* spiking_backward(void *self, float *gradients, size_t time_step) {
    SpikingLayer *layer = (SpikingLayer *)self;

    for (size_t i = 0; i < layer->num_neurons; i++) {
        LIFNeuron *neuron = (LIFNeuron *)layer->neurons[i];
        
        // Load membrane potential for this time step
        if (layer->membrane_history) {
            neuron->base.v = layer->membrane_history[time_step * layer->num_neurons + i];
            neuron->base.spiked = layer->spike_history[time_step * layer->num_neurons + i];
        }
        
        // ATAN surrogate gradient (matches snnTorch default)
        float spike_derivative = 1.0f / (1.0f + neuron->base.v * neuron->base.v);
        
        // Combine with incoming gradients
        layer->spike_gradients[i] = gradients[i] * spike_derivative;
        
        // Propagate gradient through membrane potential
        layer->base.input_gradients[i] = layer->spike_gradients[i] * neuron->beta;
    }

    return layer->base.input_gradients;
}

void spiking_reset_spike_counts(void *self) {
    SpikingLayer *layer = (SpikingLayer *)self;
    for (size_t i = 0; i < layer->num_neurons; i++) {
        LIFNeuron *neuron = (LIFNeuron *)layer->neurons[i];
        neuron->spike_count = 0;
    }
}

void spiking_free(SpikingLayer *layer) {
    free(layer->neurons);
    free(layer->spike_gradients);
    free(layer->base.input_gradients);
}
