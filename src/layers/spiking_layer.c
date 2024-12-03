#include <stdlib.h>
#include <stdio.h>

#include "../../include/layers/spiking_layer.h"

// Initialize the spiking layer with given neurons
void spiking_initialize(SpikingLayer *layer, size_t num_neurons, ModelBase **neuron_models) {
    layer->num_neurons = num_neurons;
    layer->neurons = (ModelBase **)malloc(num_neurons * sizeof(ModelBase *));
    layer->output_spikes = (float *)malloc(num_neurons * sizeof(float));

    // Assign neuron models to the layer
    for (size_t i = 0; i < num_neurons; i++) {
        layer->neurons[i] = neuron_models[i];
    }
}

// Forward pass: update all neurons in the layer and record spikes
void spiking_forward(void *self, float *input, size_t input_size) {
    printf("Performing Spiking Layer forward pass...\n");
    SpikingLayer* layer = (SpikingLayer*)self;

    for (size_t i = 0; i < layer->num_neurons; i++) {
        // Call the polymorphic update function for each neuron
        layer->neurons[i]->update_neuron(layer->neurons[i], input[i]);

        // Check if the neuron spiked and record it
        if (layer->neurons[i]->v >= layer->neurons[i]->v_threshold) {
            layer->output_spikes[i] = 1.0f;
            printf("Neuron %lu spiked!\n", i);
        } else {
            layer->output_spikes[i] = 0.0f;
        }
    }
}

// Free allocated memory
void spiking_free(SpikingLayer *layer) {
    free(layer->neurons);
    free(layer->output_spikes);
}
