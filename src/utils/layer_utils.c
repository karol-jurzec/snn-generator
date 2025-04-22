#include "../../include/utils/layer_utils.h"

size_t calculate_output_dim(size_t input_dim, int kernel_size, int stride, int padding) {
    return ((input_dim - kernel_size + 2 * padding) / stride) + 1;
}

void initialize_biases(float *biases, size_t size, int fan_in) {
    float limit = 1.0f / sqrtf(fan_in);
    for (size_t i = 0; i < size; i++) {
        float rand_val = (float)rand() / RAND_MAX; // Generates a value in [0, 1]
        biases[i] = (2.0f * rand_val - 1.0f) * limit; // Scale to [-limit, limit]
    }
}
