#include "../../include/models/lif_neuron.h"
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <string.h>

// Windows-compatible aligned allocation
#ifdef _WIN32
#include <malloc.h>
#define ALIGNED_ALLOC(alignment, size) _aligned_malloc(size, alignment)
#define ALIGNED_FREE _aligned_free
#else
#define ALIGNED_ALLOC(alignment, size) aligned_alloc(alignment, size)
#define ALIGNED_FREE free
#endif

void lif_initialize(LIFLayer *layer, size_t num_neurons, float v_rest, float threshold, float v_reset, float beta) {
    layer->num_neurons = num_neurons;
    
    // Allocate aligned memory for SIMD
    layer->v = (float *)ALIGNED_ALLOC(64, num_neurons * sizeof(float));
    layer->v_threshold = (float *)ALIGNED_ALLOC(64, num_neurons * sizeof(float));
    layer->v_rest = (float *)ALIGNED_ALLOC(64, num_neurons * sizeof(float));
    layer->v_reset = (float *)ALIGNED_ALLOC(64, num_neurons * sizeof(float));
    layer->beta = (float *)ALIGNED_ALLOC(64, num_neurons * sizeof(float));
    layer->spiked = (int *)ALIGNED_ALLOC(64, num_neurons * sizeof(int));
    layer->spike_count = (int *)ALIGNED_ALLOC(64, num_neurons * sizeof(int));
    layer->total_spike_counts = (int *)ALIGNED_ALLOC(64, num_neurons * sizeof(int));

    if (!layer->v || !layer->v_threshold || !layer->v_rest || 
        !layer->v_reset || !layer->beta || !layer->spiked || !layer->spike_count) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Initialize values
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < num_neurons; i++) {
        layer->v[i] = v_rest;
        layer->v_threshold[i] = threshold;
        layer->v_rest[i] = v_rest;
        layer->v_reset[i] = v_reset;
        layer->beta[i] = beta;
        layer->spiked[i] = 0;
        layer->spike_count[i] = 0;
        layer->total_spike_counts[i] = 0;
    }
}

void lif_update(void *self, float *input_current, size_t num_neurons) {
    LIFLayer *layer = (LIFLayer *)self;
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < num_neurons; i += 16) {
        if (i + 16 <= num_neurons) {
            // SIMD processing for 8 neurons at a time
            __m256 v = _mm256_load_ps(&layer->v[i]);
            __m256 beta = _mm256_load_ps(&layer->beta[i]);
            __m256 input = _mm256_load_ps(&input_current[i]);
            __m256 threshold = _mm256_load_ps(&layer->v_threshold[i]);
            
            // Update membrane potential: v = beta * v + input
            v = _mm256_fmadd_ps(beta, v, input);
            _mm256_store_ps(&layer->v[i], v);
            
            // Check for spikes
            __m256 mask = _mm256_cmp_ps(v, threshold, _CMP_GE_OQ);
            int mask_int = _mm256_movemask_ps(mask);
            
            // Handle spikes
            for (int j = 0; j < 8; j++) {
                if (mask_int & (1 << j)) {
                    layer->v[i + j] -= layer->v_threshold[i + j];  // subtractive reset
                    layer->spiked[i + j] = 1;
                    layer->spike_count[i + j]++;
                    layer->total_spike_counts[i + j]++;
                } else {
                    layer->spiked[i + j] = 0;
                }
            }
        } else {
            // Scalar fallback for remaining neurons
            for (size_t j = i; j < num_neurons; j++) {
                layer->v[j] = layer->beta[j] * layer->v[j] + input_current[j];
                
                if (layer->v[j] >= layer->v_threshold[j]) {
                    layer->v[j] -= layer->v_threshold[j];  // subtractive reset
                    layer->spiked[j] = 1;
                    layer->spike_count[j]++;
                    layer->total_spike_counts[i + j]++;
                } else {
                    layer->spiked[j] = 0;
                }
            }
        }
    }
}

void lif_free(LIFLayer *layer) {
    ALIGNED_FREE(layer->v);
    ALIGNED_FREE(layer->v_threshold);
    ALIGNED_FREE(layer->v_rest);
    ALIGNED_FREE(layer->v_reset);
    ALIGNED_FREE(layer->beta);
    ALIGNED_FREE(layer->spiked);
    ALIGNED_FREE(layer->spike_count);
}