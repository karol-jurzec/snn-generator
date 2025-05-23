#ifndef LIF_NEURON_H
#define LIF_NEURON_H

#include "model_base.h"
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// Structure of Arrays (SoA) for better SIMD performance
typedef struct {
    float *v;               // Membrane potential
    float *v_threshold;
    float *v_rest;
    float *v_reset;
    float *beta;
    int *spiked;
    int *spike_count;
    int *total_spike_counts;
    size_t num_neurons;
} LIFLayer;

void lif_initialize(LIFLayer *layer, size_t num_neurons, float v_rest, float threshold, float v_reset, float beta);
void lif_update(void *self, float *input_current, size_t num_neurons);
void lif_free(LIFLayer *layer);

#endif // LIF_NEURON_H