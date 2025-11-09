#include "../../include/models/izhikevich_neuron.h"
#include <stdio.h>

void izhikevich_initialize(IzhikevichNeuron *neuron, float a, float b, float c, float d, float v_rest) {
    neuron->base.v = v_rest;
    neuron->u = b * v_rest;
    neuron->a = a;
    neuron->b = b;
    neuron->c = c;
    neuron->d = d;
    neuron->spiked = 0;
    neuron->base.update_neuron = izhikevich_update;
}

void izhikevich_update(void *self, float input_current) {
    IzhikevichNeuron *neuron = (IzhikevichNeuron *)self;

    if (neuron->base.v >= 30.0f) {
        neuron->base.v = neuron->c;
        neuron->u += neuron->d;
        neuron->spiked = 1;
    } else {
        float dv = 0.04f * neuron->base.v * neuron->base.v + 5.0f * neuron->base.v + 140.0f - neuron->u + input_current;
        float du = neuron->a * (neuron->b * neuron->base.v - neuron->u);
        neuron->base.v += dv;
        neuron->u += du;
        neuron->spiked = 0;
    }
}
