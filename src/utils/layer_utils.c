#include "../../include/utils/layer_utils.h"

size_t calculate_output_dim(size_t input_dim, int kernel_size, int stride, int padding) {
    return ((input_dim - kernel_size + 2 * padding) / stride) + 1;
}
