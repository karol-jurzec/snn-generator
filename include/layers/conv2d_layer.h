#ifndef CONV2D_LAYER_H
#define CONV2D_LAYER_H

#include <stddef.h>
#include "layer_base.h"
#include <openblas/cblas.h>


typedef struct {
    LayerBase base;          
    int input_dim;           
    int in_channels;         // EFFECTIVE in_channels (po pruning może być mniejsze)
    int out_channels;        // EFFECTIVE out_channels (po pruning może być mniejsze)
    int kernel_size;         
    int stride;              
    int padding;             
    float *weights;          
    float *biases;   
    
    // persistent forward buffers and cache metadata
    float *tmp_im_pruned;   // size: max_effective_IC * H * H
    float *data_col;        // size: max_effective_IC * K2 * N
    float *weight_mat;      // size: max_effective_OC * max_effective_IC * K2
    float *out_mat;         // size: max_effective_OC * N

    // caching info to detect when to reallocate / rebuild
    int cache_IC;
    int cache_OC;
    int cache_K2;
    int cache_N;
    bool weight_mat_valid;
    
    // ORIGINAL DIMENSIONS (przed pruning)
    int original_in_channels;   // Oryginalna liczba input channels
    int original_out_channels;  // Oryginalna liczba output channels
    
    // DYNAMIC CHANNEL MAPPING (NULL = no pruning)
    int *in_active_channels_idx;   // [0,1,3,5,...] - indeksy aktywnych input channels
    int *out_active_channels_idx;  // [0,2,4,6,...] - indeksy aktywnych output channels
} Conv2DLayer;


void conv2d_initialize(Conv2DLayer *layer, int in_channels, int out_channels, int kernel_size, int stride, int padding, int input_dim);
void conv2d_forward(void *self, float *input, size_t input_size, size_t time_step);
void conv2d_free(Conv2DLayer *layer);
void conv2d_invalidate_weight_mat(Conv2DLayer *L);
void build_weight_mat_if_needed(Conv2DLayer *L, int IC, int OC, int K2);
void ensure_forward_buffers(Conv2DLayer *L, int IC, int OC, int K2, int N);



#endif // CONV2D_LAYER_H
