#include <stdlib.h>
#include <stdio.h>

#include "../../include/layers/spiking_layer.h"

void spiking_initialize(SpikingLayer *layer, size_t num_neurons, ModelBase **neuron_models) {
    layer->base.forward = spiking_forward;
    layer->num_neurons = num_neurons;
    layer->neurons = (ModelBase **)malloc(num_neurons * sizeof(ModelBase *));
    layer->base.output = (float *)malloc(num_neurons * sizeof(float));
    layer->base.output_size = num_neurons;

    for (size_t i = 0; i < num_neurons; i++) {
        layer->neurons[i] = neuron_models[i];
    }
}

void spiking_forward(void *self, float *input, size_t input_size) {
    printf("Performing Spiking Layer forward pass...\n");
    SpikingLayer* layer = (SpikingLayer*)self;

    for (size_t i = 0; i < input_size; i++) {
        if (i >= layer->num_neurons) {
            fprintf(stderr, "Error: input_size exceeds number of neurons\n");
            return;
        }

        if (layer->neurons[i] == NULL) {
            fprintf(stderr, "Error: neuron %lu is not initialized\n", i);
            return;
        }

        //printf("Updating neuron %lu with input %f\n", i, input[i]);
        layer->neurons[i]->update_neuron(layer->neurons[i], input[i]);

        if (layer->neurons[i]->v >= layer->neurons[i]->v_threshold) {
            layer->base.output[i] = 1.0f;
            printf("Neuron %lu spiked!\n", i);
        } else {
            layer->base.output[i] = 0.0f;
        }
    }

    printf("Neuron layer values: \n");
    for(int i = 0; i < 10; ++i) {
        printf("Neuron %d: %f\n", i, layer->neurons[i]->v);
    }
}

void spiking_free(SpikingLayer *layer) {
    free(layer->neurons);
    free(layer->output_spikes);
}
