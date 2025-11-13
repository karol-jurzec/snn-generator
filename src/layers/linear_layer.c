// linear_layer.c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include <openblas/cblas.h>

#include "../../include/layers/linear_layer.h"
#include "../../include/utils/layer_utils.h"

void he_kaiming_uniform_init_linear(float *weights, size_t size, int fan_in) {
    float limit = sqrtf(6.0f / fan_in);
    for (size_t i = 0; i < size; i++) {
        float rand_val = (float)rand() / RAND_MAX; // [0, 1]
        weights[i] = (2.0f * rand_val - 1.0f) * limit; // Scale to [-limit, limit]
    }
}

void linear_initialize(LinearLayer *layer, size_t in_features, size_t out_features) {
    layer->base.layer_type = LAYER_LINEAR;
    layer->base.forward = linear_forward;
    layer->base.is_spiking = false;
    layer->base.num_inputs = in_features;

    layer->in_features = (int)in_features;
    layer->out_features = (int)out_features;

    layer->base.inputs = (float*)malloc(in_features * sizeof(float));
    if (!layer->base.inputs) { fprintf(stderr, "linear_initialize: alloc inputs failed\n"); exit(1); }

    layer->base.weights = (float *)malloc(in_features * out_features * sizeof(float));
    if (!layer->base.weights) { fprintf(stderr, "linear_initialize: alloc weights failed\n"); exit(1); }
    layer->base.num_weights = in_features * out_features;

    layer->biases = (float *)malloc(out_features * sizeof(float));
    if (!layer->biases) { fprintf(stderr, "linear_initialize: alloc biases failed\n"); exit(1); }

    layer->base.output = (float *)malloc(out_features * sizeof(float));
    if (!layer->base.output) { fprintf(stderr, "linear_initialize: alloc output failed\n"); exit(1); }
    layer->base.output_size = out_features;

    // initialize weights and biases
    he_kaiming_uniform_init_linear(layer->base.weights, layer->base.num_weights, (int)in_features);
    for (size_t i = 0; i < out_features; ++i) layer->biases[i] = 0.0f;

    // pruning arrays default to NULL (no pruning)
    layer->in_active_channels_idx = NULL;
    layer->out_active_channels_idx = NULL;

    // persistent compact weight matrix (used when pruning present)
    layer->weight_mat = NULL;
    layer->cache_IC = -1;
    layer->cache_OC = -1;
    layer->weight_mat_valid = false;
}

/* Build compact weight matrix (OC x IC) row-major.
   If no pruning is active, this will simply copy the original weight matrix
   into weight_mat in the same layout (out-major rows).
   This function sets weight_mat_valid = true when done.
*/
static void linear_build_weight_mat_if_needed(LinearLayer *L) {
    int IC = L->in_features;
    int OC = L->out_features;

    if (L->weight_mat_valid && L->cache_IC == IC && L->cache_OC == OC) return;

    // ensure allocation
    size_t elems = (size_t)OC * IC;
    if (!L->weight_mat) {
        L->weight_mat = (float *)malloc(elems * sizeof(float));
        if (!L->weight_mat) { fprintf(stderr, "linear: alloc weight_mat failed\n"); exit(1); }
    }

    // If no pruning, base.weights layout is expected as row-major [out][in]
    // If pruning indices exist, we gather rows/cols accordingly.
    for (int o_idx = 0; o_idx < OC; ++o_idx) {
        int o_actual = L->out_active_channels_idx ? L->out_active_channels_idx[o_idx] : o_idx;
        float *dst_row = L->weight_mat + (size_t)o_idx * IC;
        if (L->in_active_channels_idx) {
            // gather selected input columns
            for (int i_idx = 0; i_idx < IC; ++i_idx) {
                int i_actual = L->in_active_channels_idx[i_idx];
                dst_row[i_idx] = L->base.weights[(size_t)o_actual * L->in_features + i_actual];
            }
        } else {
            // copy contiguous in-features row
            memcpy(dst_row, L->base.weights + (size_t)o_actual * L->in_features, (size_t)IC * sizeof(float));
        }
    }

    L->cache_IC = IC;
    L->cache_OC = OC;
    L->weight_mat_valid = true;
}

/* Call this when pruning indices change to force rebuild of compact weight matrix */
void linear_invalidate_weight_mat(LinearLayer *L) {
    L->weight_mat_valid = false;
}

/* Forward uses cblas_sgemv:
   y = A * x + b
   where A is weight_mat (OC x IC), x is input (IC)
*/
void linear_forward(void *self, float *input, size_t input_size, size_t time_step) {
    (void)time_step;
    LinearLayer *L = (LinearLayer*)self;

    // copy inputs into base.inputs (keeps same behavior as before)
    memcpy(L->base.inputs, input, input_size * sizeof(float));

    int IC = L->in_features;
    int OC = L->out_features;

    // Build compact weight matrix if pruning is active or if not valid
    linear_build_weight_mat_if_needed(L);

    // If no pruning and weight_mat is not allocated, we can use base.weights directly with cblas.
    float *A = L->weight_mat ? L->weight_mat : L->base.weights;

    // If pruning on inputs/out, we provided compact matrix, else A points to row-major base.weights.
    // Use cblas_sgemv: CblasRowMajor, CblasNoTrans, M=OC rows, N=IC cols
    // alpha=1, beta=1 (we'll add bias in a second — set beta=0 to write directly)
    // We'll write result into L->base.output using beta=0 then add biases
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                OC, IC,
                1.0f,
                A, IC,
                input, 1,
                0.0f,
                L->base.output, 1);

    // add biases
    for (int o = 0; o < OC; ++o) {
        int o_actual = L->out_active_channels_idx ? L->out_active_channels_idx[o] : o;
        // If pruning is active and output indices scatter into original slots is desired,
        // we currently keep output dense: base.output is OC-sized array (dense). If you
        // prefer scattering into original-slot-sized array, adapt accordingly.
        L->base.output[o] += L->biases[o_actual];
    }
}

void linear_free(LinearLayer *layer) {
    if (!layer) return;
    if (layer->base.weights) free(layer->base.weights);
    if (layer->biases) free(layer->biases);
    if (layer->base.output) free(layer->base.output);
    if (layer->base.inputs) free(layer->base.inputs);

    if (layer->in_active_channels_idx) free(layer->in_active_channels_idx);
    if (layer->out_active_channels_idx) free(layer->out_active_channels_idx);

    if (layer->weight_mat) free(layer->weight_mat);
}
