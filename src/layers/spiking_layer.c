#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../../include/layers/spiking_layer.h"

// Windows-compatible aligned allocation
#ifdef _WIN32
#include <malloc.h>
#define ALIGNED_ALLOC(alignment, size) _aligned_malloc(size, alignment)
#define ALIGNED_FREE _aligned_free
#else
#define ALIGNED_ALLOC(alignment, size) aligned_alloc(alignment, size)
#define ALIGNED_FREE free
#endif

void spiking_initialize(SpikingLayer *layer, size_t num_neurons, float v_rest, float threshold, float v_reset, float beta) {
    layer->base.layer_type = LAYER_SPIKING;
    layer->base.is_spiking = true;
    layer->base.forward = spiking_forward;
    layer->base.backward = spiking_backward;
    layer->base.zero_grad = spiking_zero_grad;
    layer->base.reset_spike_counts = spiking_reset_spike_counts;
    layer->base.num_inputs = num_neurons;
    layer->base.output_size = num_neurons;
    
    // Initialize neuron layer with SoA
    lif_initialize(&layer->neuron_layer, num_neurons, v_rest, threshold, v_reset, beta);
    
    // Allocate output arrays with aligned memory
    layer->base.output = (float *)ALIGNED_ALLOC(64, num_neurons * sizeof(float));
    layer->input_gradients = (float *)ALIGNED_ALLOC(64, num_neurons * sizeof(float));
    layer->spike_gradients = (float *)ALIGNED_ALLOC(64, num_neurons * sizeof(float));
    
    if (!layer->base.output || !layer->input_gradients || !layer->spike_gradients) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    
    // Initialize to zero
    memset(layer->base.output, 0, num_neurons * sizeof(float));
    memset(layer->input_gradients, 0, num_neurons * sizeof(float));
    memset(layer->spike_gradients, 0, num_neurons * sizeof(float));
}

void spiking_forward(void *self, float *input, size_t input_size, size_t time_step) {
    (void)time_step; // Mark unused parameter
    SpikingLayer *layer = (SpikingLayer *)self;
    
    // Update neurons
    lif_update(&layer->neuron_layer, input, input_size);
    
    // Copy spikes to output
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < (size_t)layer->base.output_size; i++) {
        layer->base.output[i] = (float)layer->neuron_layer.spiked[i];
    }
}

float* spiking_backward(void *self, float *gradients, size_t time_step) {
    (void)time_step; // Mark unused parameter
    SpikingLayer *layer = (SpikingLayer *)self;
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < (size_t)layer->base.output_size; i++) {
        // ATAN surrogate gradient
        float spike_derivative = 1.0f / (1.0f + layer->neuron_layer.v[i] * layer->neuron_layer.v[i]);
        
        // Combine with incoming gradients
        layer->spike_gradients[i] = gradients[i] * spike_derivative;
        
        // Propagate gradient through membrane potential
        layer->input_gradients[i] = layer->spike_gradients[i] * layer->neuron_layer.beta[i];
    }
    
    return layer->input_gradients;
}

void spiking_zero_grad(void *self) {
    SpikingLayer *layer = (SpikingLayer *)self;
    memset(layer->spike_gradients, 0, layer->base.output_size * sizeof(float));
    memset(layer->input_gradients, 0, layer->base.output_size * sizeof(float));
}

void spiking_reset_spike_counts(void *self) {
    SpikingLayer *layer = (SpikingLayer *)self;
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < (size_t)layer->base.output_size; i++) {
        layer->neuron_layer.v[i] = layer->neuron_layer.v_rest[i];
        layer->neuron_layer.spike_count[i] = 0;
        layer->neuron_layer.spiked[i] = 0;
    }
}

void spiking_free(SpikingLayer *layer) {
    lif_free(&layer->neuron_layer);
    ALIGNED_FREE(layer->base.output);
    ALIGNED_FREE(layer->input_gradients);
    ALIGNED_FREE(layer->spike_gradients);
}