#include "../../include/models/lif_neuron.h"
#include <stdio.h>

void lif_initialize(LIFNeuron *neuron, float v_rest, float threshold, float v_reset, float beta) {
    neuron->base.v = v_rest;
    neuron->base.v_threshold = threshold;
    neuron->base.update_neuron = lif_update;
    neuron->v_rest = v_rest;
    neuron->v_reset = v_reset;
    neuron->beta = beta;
    neuron->spiked = 0;
    neuron->spike_count = 0;
}

void lif_update(void *self, float input_current) {
    LIFNeuron *neuron = (LIFNeuron *)self;
    neuron->base.v = neuron->beta * neuron->base.v + input_current;

    printf("LIF Neuron membrane potential: %f\n", neuron->base.v);

    if (neuron->base.v >= neuron->base.v_threshold) {
        printf("LIF Neuron spiked!\n");
        neuron->spiked = 1;
        neuron->base.v = neuron->v_reset;
        neuron->spike_count++;
    } else {
        neuron->spiked = 0;
    }
}
