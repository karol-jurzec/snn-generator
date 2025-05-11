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

void im2col(float* data_im, int channels, int height, int width,
    int kernel_size, int padding, int stride,
    float* data_col, int output_h, int output_w) {
int c, h, w;
int col_index = 0;

for (c = 0; c < channels; ++c) {
    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            for (h = 0; h < output_h; ++h) {
                for (w = 0; w < output_w; ++w) {
                    int im_row = h * stride + ky - padding;
                    int im_col = w * stride + kx - padding;
                    int im_index = c * height * width + im_row * width + im_col;
                    if (im_row >= 0 && im_row < height && im_col >= 0 && im_col < width) {
                        data_col[col_index++] = data_im[im_index];
                    } else {
                        data_col[col_index++] = 0.0f;
                    }
                    }
                }
            }
        }
    }
}

void col2im(const float *data_col,
    int channels, int height, int width,
    int kernel_size, int padding, int stride,
    float *data_im, int output_h, int output_w)
{
    // zero the output buffer
    int img_size = channels * height * width;
    for(int i=0; i<img_size; i++) data_im[i] = 0.0f;

    int c, ky, kx, h, w;
    int col_index = 0;
    for (c = 0; c < channels; ++c) {
        for (ky = 0; ky < kernel_size; ++ky) {
            for (kx = 0; kx < kernel_size; ++kx) {
                for (h = 0; h < output_h; ++h) {
                    for (w = 0; w < output_w; ++w) {
                        int im_row = h * stride + ky - padding;
                        int im_col = w * stride + kx - padding;
                        if (im_row >= 0 && im_row < height && im_col >= 0 && im_col < width) {
                            int im_idx = c*height*width + im_row*width + im_col;
                            data_im[im_idx] += data_col[col_index];
                        }
                        col_index++;
                    }
                }
            }
        }
    }
}
