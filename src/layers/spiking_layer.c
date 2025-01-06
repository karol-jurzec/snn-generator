#include <stdlib.h>
#include <stdio.h>

#include "../../include/layers/spiking_layer.h"

void spiking_initialize(SpikingLayer *layer, size_t num_neurons, ModelBase **neuron_models) {
    layer->base.forward = spiking_forward;
    layer->base.backward = spiking_backward;
    layer->num_neurons = num_neurons;
    layer->neurons = (ModelBase **)malloc(num_neurons * sizeof(ModelBase *));
    layer->base.output = (float *)malloc(num_neurons * sizeof(float));
    layer->base.output_size = num_neurons;

    for (size_t i = 0; i < num_neurons; i++) {
        layer->neurons[i] = neuron_models[i];
    }

    layer->spike_gradients = (float *)malloc(num_neurons * sizeof(float));
    layer->base.input_gradients = (float *)malloc(num_neurons * sizeof(float));
}

void spiking_forward(void *self, float *input, size_t input_size) {
    printf("Input for spiking layer:\n");
    for(int i = 0; i < 10; ++i) {
        printf("input[%d] = %f\n", i, input[i]);
    }

    //printf("Performing Spiking Layer forward pass...\n");
    SpikingLayer* layer = (SpikingLayer*)self;

    for (size_t i = 0; i < layer->num_neurons; i++) {
        layer->neurons[i]->update_neuron(layer->neurons[i], input[i]);
        layer->base.output[i] = (layer->neurons[i]->spiked == 1) ? 1.0f : 0.0f;
    }

}

// Backward pass for Spiking Layer
void spiking_backward(void *self, float *gradients) {
    SpikingLayer *layer = (SpikingLayer *)self;

    for (size_t i = 0; i < layer->num_neurons; i++) {
        float membrane_potential = layer->neurons[i]->v;
        float spike_grad = (1.0f - membrane_potential * membrane_potential); // Example: tanh derivative
        layer->spike_gradients[i] = gradients[i] * spike_grad;

        layer->base.input_gradients[i] = layer->spike_gradients[i];
    }
}

void spiking_free(SpikingLayer *layer) {
    free(layer->neurons);
    free(layer->spike_gradients);
    free(layer->base.input_gradients);
}
