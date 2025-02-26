#ifndef LAYER_BASE_H
#define LAYER_BASE_H

#include <stddef.h>

// Macro for polymorphic forward function in layers
#define LAYER_FORWARD_FUNC(name) void (*name)(void *self, float *input, size_t input_size)
#define LAYER_BACKWARD_FUNC(name) void (*name)(void *self, float *gradients)
#define UPDATE_WEIGHTS_FUNC(name) void (*name)(void *self, float learning_rate)
#define RESET_SPIKE_COUNTS_FUNC(name) void (*name)(void *self) 

// Base structure for layers
typedef struct {
    LAYER_FORWARD_FUNC(forward);  // Forward function pointer for layers
    LAYER_BACKWARD_FUNC(backward);  // Backward function pointer for layers
    UPDATE_WEIGHTS_FUNC(update_weights);  // Update weights function pointer for layers
    RESET_SPIKE_COUNTS_FUNC(reset_spike_counts);  // Reset spike counts function pointer for spiking layers
    float *output;                // Output buffer for layers
    float *input_gradients;       // Input gradients for backpropagation
    
    float *weight_gradients; // Gradient of weights
    float *bias_gradients;   // Gradient of biases

    float *weights;
    int num_weights;
    int output_size;
} LayerBase;

#endif // LAYER_BASE_H