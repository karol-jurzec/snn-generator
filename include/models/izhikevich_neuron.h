#ifndef IZHIKEVICH_NEURON_H
#define IZHIKEVICH_NEURON_H

#include "model_base.h"

// Structure for Izhikevich neuron
typedef struct {
    ModelBase base;  // Inherits ModelBase
    float u;         // Recovery variable
    float a, b, c, d; // Model parameters
    int spiked;
} IzhikevichNeuron;

void izhikevich_initialize(IzhikevichNeuron *neuron, float a, float b, float c, float d, float v_rest);
void izhikevich_update(void *self, float input_current);

#endif // IZHIKEVICH_NEURON_H
