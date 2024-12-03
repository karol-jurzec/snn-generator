#ifndef LIF_NEURON_H
#define LIF_NEURON_H

#include "model_base.h"

typedef struct {
    ModelBase base;   // Inherits ModelBase
    float v_rest;
    float v_threshold;
    float v_reset;
    float beta;
    int spiked;
    int spike_count;
} LIFNeuron;

void lif_initialize(LIFNeuron *neuron, float v_rest, float threshold, float v_reset, float beta);
void lif_update(void *self, float input_current);

#endif // LIF_NEURON_H
