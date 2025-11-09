#include <stdlib.h>
#include <stdio.h>

#include "../../include/layers/spiking_layer.h"
#include "../../include/models/lif_neuron.h"

void spiking_initialize(SpikingLayer *layer, size_t num_neurons, ModelBase **neuron_models) {
    layer->base.layer_type = LAYER_SPIKING;
    layer->base.is_spiking = true;
    layer->base.forward = spiking_forward;
    layer->base.reset_spike_counts = spiking_reset_spike_counts;
    layer->base.num_inputs= num_neurons;
    layer->num_neurons = num_neurons;
    layer->neurons = (ModelBase **)malloc(num_neurons * sizeof(ModelBase *));
    layer->base.inputs = (float *)malloc(num_neurons * sizeof(float));
    layer->base.output = (float *)malloc(num_neurons * sizeof(float));
    layer->total_spikes = (int *)calloc(num_neurons,  sizeof(int));

    layer->base.output_size = num_neurons;

    for (size_t i = 0; i < num_neurons; i++) {
        layer->neurons[i] = neuron_models[i];
    }
}

void spiking_forward(void *self, float *input, size_t input_size, size_t time_step) {
    SpikingLayer *layer = (SpikingLayer *)self;
    
    memcpy(layer->base.inputs, input, input_size * sizeof(float));

    for (size_t i = 0; i < layer->num_neurons; i++) {
        layer->neurons[i]->update_neuron(layer->neurons[i], input[i]);
        layer->base.output[i] = layer->neurons[i]->spiked;
        layer->total_spikes[i] += layer->base.output[i];
    }
}

void spiking_reset_spike_counts(void *self) {
    SpikingLayer *layer = (SpikingLayer *)self;
    for (size_t i = 0; i < layer->num_neurons; i++) {
        LIFNeuron *neuron = (LIFNeuron *)layer->neurons[i];
        neuron->base.v = 0;    
        neuron->spike_count = 0;
        neuron->base.spiked = 0;
    }
}

void spiking_free(SpikingLayer *layer) {
    free(layer->neurons);
}
