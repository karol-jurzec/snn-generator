#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float v;            // Membrane potential
    float v_rest;       // Resting potential
    float v_threshold;  // Firing threshold
    float v_reset;      // Reset potential
    float beta;         // Decay factor (calculated from tau)
    float i;            // Input current
    int spiked;         // Binary flag indicating if neuron has spiked
    int spike_count;    // Spike counter
} LIFNeuron;

// Initialize single LIF neuron with default parameters
void initialize_neuron(LIFNeuron *neuron, float v_rest, float threshold, float v_reset, float beta) {
    neuron->v = v_rest;            
    neuron->v_rest = v_rest;
    neuron->v_threshold = threshold;
    neuron->v_reset = v_reset;
    neuron->beta= beta; 
    neuron->spiked = 0;
    neuron->spike_count = 0;
}

// Update neuron membrane potential based on input current
// Apply LIF neuron update formula with decay and input current
// neuron->v = neuron->v * neuron->decay + input_current * (1 - neuron->decay);
void update_neuron(LIFNeuron *neuron, float input_current) {
    neuron->v = neuron->beta * neuron->v  + input_current;
    printf("Neuron membrane potential: %f \n", neuron->v);

    if (neuron->v >= neuron->v_threshold) {
        printf("Neuron has spiked!\n");
        neuron->spiked = 1;
        neuron->v = neuron->v_reset;  
        neuron->spike_count++;
    } else {
        neuron->spiked = 0;
    }
}

//void update_neuron(LIFNeuron* neuron, float dt) {
    //neuron->V += (-(neuron->V - neuron->V_rest) + neuron->I) * (dt / neuron->tau);


//    if (neuron->v >= neuron->v_threshold) {
//        neuron->v = neuron->v_reset;  // Reset potential after firing
//    }
//}