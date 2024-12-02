#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "model_base.c"

#ifndef __cplusplus
#include <stdbool.h>
#endif

//#define debug true

typedef struct {
    float v_rest;       // Resting potential
    float v_threshold;  // Firing threshold
    float v_reset;      // Reset potential
    float beta;         // Decay factor (calculated from tau)
    float i;            // Input current
    int spiked;         // Binary flag indicating if neuron has spiked
    int spike_count;    // Spike counter

    ModelBase model_base; 
} LIFNeuron;

// Update neuron membrane potential based on input current
// Apply LIF neuron update formula with decay and input current
// neuron->v = neuron->v * neuron->decay + input_current * (1 - neuron->decay);
void update_leaky(void *self, float input_current) {
    LIFNeuron* neuron = (LIFNeuron*)self; 
    neuron->model_base.v = neuron->beta * neuron->model_base.v  + input_current;

    #ifdef debug
    printf("Neuron membrane potential: %f \n", neuron->v);
    #endif

    if (neuron->model_base.v >= neuron->v_threshold) {
        #ifdef deubg
        printf("Neuron has spiked!\n");
        #endif
        neuron->spiked = 1;
        neuron->model_base.v = neuron->v_reset;  
        neuron->spike_count++;
    } else {
        neuron->spiked = 0;
    }
}

// initialize single LIF neuron with default parameters
void initialize_neuron(LIFNeuron *neuron, float v_rest, float threshold, float v_reset, float beta) {
    neuron->v_rest = v_rest;
    neuron->v_threshold = threshold;
    neuron->v_reset = v_reset;
    neuron->beta= beta; 
    neuron->spiked = 0;
    neuron->spike_count = 0;

    neuron->model_base.v = v_rest;            
    neuron->model_base.update_neuron = update_leaky;
}

// logging function for membrane potential values
void log_membrane_potential(FILE *file, float membrane_potential, int dt) {
    if (file != NULL) {
        fprintf(file, "%d, %f\n", dt, membrane_potential);
    }
}

