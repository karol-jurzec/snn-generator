#ifndef LAYER_BASE_H
#define LAYER_BASE_H

#include <stddef.h>

// Macro for polymorphic forward function in layers
#define LAYER_FORWARD_FUNC(name) void (*name)(void *self, float *input, size_t input_size)

// Base structure for layers
typedef struct {
    LAYER_FORWARD_FUNC(forward);  // Forward function pointer for layers
} LayerBase;

#endif // LAYER_BASE_H