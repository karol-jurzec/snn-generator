#ifndef FLATTEN_LAYER_H
#define FLATTEN_LAYER_H

#include <stddef.h>
#include "layer_base.h"

typedef struct {
    LayerBase base;          // Inherits LayerBase for polymorphic forward
    float *output;           // Output buffer (1D flattened output)
    size_t output_size;      // Size of the flattened output
} FlattenLayer;

// Function declarations
void flatten_initialize(FlattenLayer *layer, size_t input_size);
void flatten_forward(void *self, float *input, size_t input_size, size_t time_step);

void flatten_free(FlattenLayer *layer);
#endif // FLATTEN_LAYER_H
