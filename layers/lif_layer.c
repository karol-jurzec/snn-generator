#include "../models/lif_neuron.c"

#define NUM_NEURONS 5 

typedef struct {
    LIFNeuron neurons[NUM_NEURONS];
} LIFLayer;

void initalize_layer(LIFLayer *layer, float v_rest, float threshold, float v_reset, float beta) {
    for(int i = 0; i < NUM_NEURONS; ++i) {
        initialize_neuron(&layer->neurons[i], v_rest, threshold, v_reset, beta);
    }
}

void update_layer(LIFLayer *layer, float input_currents[NUM_NEURONS]) {
    for(int i = 0; i < NUM_NEURONS; ++i) {
        update_neuron(&layer->neurons[i], input_currents[i]);
    }
}

void reset_spike_counts(LIFLayer *layer) {
    for(int i = 0; i < NUM_NEURONS; ++i) {
        layer->neurons[i].spike_count = 0;
    }
}
