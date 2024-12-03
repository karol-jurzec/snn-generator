#ifndef MODEL_BASE_H
#define MODEL_BASE_H

#include <stddef.h>

// Macro for abstract function inheritance
#define MODEL_UPDATE_FUNC(name) void (*name)(void *self, float input_current)

typedef struct {
    float v;                       // Membrane potential
    float v_threshold;
    MODEL_UPDATE_FUNC(update_neuron);  // Polymorphic update function
} ModelBase;

#endif 
