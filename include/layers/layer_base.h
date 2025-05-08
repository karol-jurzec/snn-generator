#ifndef LAYER_BASE_H
#define LAYER_BASE_H

#include <stddef.h>
#include <stdbool.h> 

// Macro for polymorphic forward function in layers
#define LAYER_FORWARD_FUNC(name) void (*name)(void *self, float *input, size_t input_size, size_t time_step)
#define LAYER_BACKWARD_FUNC(name) float* (*name)(void *self, float *gradients, size_t time_step)
#define UPDATE_WEIGHTS_FUNC(name) void (*name)(void *self, float learning_rate)
#define RESET_SPIKE_COUNTS_FUNC(name) void (*name)(void *self) 
#define ZERO_GRAD_FUNC(name) void (*name)(void *self)


// Enum to define different layer types
typedef enum {
    LAYER_CONV2D,
    LAYER_LINEAR,
    LAYER_MAXPOOL2D, 
    LAYER_FLATTEN, 
    LAYER_SPIKING
} LayerType;

// Base structure for layers
typedef struct {
    LAYER_FORWARD_FUNC(forward);  // Forward function pointer for layers
    LAYER_BACKWARD_FUNC(backward);  // Backward function pointer for layers
    UPDATE_WEIGHTS_FUNC(update_weights);  // Update weights function pointer for layers
    RESET_SPIKE_COUNTS_FUNC(reset_spike_counts);  // Reset spike counts function pointer for spiking layers
    ZERO_GRAD_FUNC(zero_grad);  // Zero gradients function pointer for layers

    float *output;           // Output buffer for layers
    float* output_history;  // Array of outputs for all time steps
    float* grad_history;    // Array of gradients for all time steps
    size_t time_steps;      // Number of time steps (TIME_BINS)

    float *input_gradients;       // Input gradients for backpropagation
    float *weight_gradients; // Gradient of weights
    float *bias_gradients;   // Gradient of biases

    float *weights;
    float *inputs;
    int num_weights;
    int num_inputs;
    int output_size;
    bool is_spiking;

    LayerType layer_type;  // Store layer type (Conv2D, Linear, etc.)

} LayerBase;

#endif // LAYER_BASE_H