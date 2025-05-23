    #include <stdlib.h>
    #include <stdio.h>
    #include <string.h>
    #include <math.h>

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
        layer->base.backward = linear_backward;
        layer->base.zero_grad = linear_zero_grad;  // Assign function pointer
        layer->base.update_weights = linear_update_weights;
        layer->base.num_inputs = in_features;
        
        layer->in_features = in_features;
        layer->out_features = out_features;

        layer->base.weights = (float *)malloc(in_features * out_features * sizeof(float));
        layer->base.num_weights = in_features * out_features;
        layer->biases = (float *)malloc(out_features * sizeof(float));
        layer->base.output = (float *)malloc(out_features * sizeof(float));
        layer->base.output_size = out_features;

        // weights and bias initalization

        //he_kaiming_uniform_init_linear(layer->base.weights, in_features * out_features, in_features);
        //initialize_biases(layer->biases, out_features, in_features);
        //layer->base.inputs = (float*)malloc(in_features * sizeof(float));
       // layer->base.weight_gradients = (float *)malloc(layer->base.num_weights * sizeof(float));
        //layer->base.bias_gradients = (float *)malloc(out_features * sizeof(float));
       // layer->base.input_gradients = (float *)malloc(in_features * sizeof(float));
    }

    void linear_forward(void *self, float *input, size_t input_size, size_t time_step) {
        LinearLayer *L = (LinearLayer*)self;
        // Save the raw input for backward
        //memcpy(L->base.inputs, input, input_size * sizeof(float));

        // y = W * x + b
        // W: [out_features x in_features], x: [in_features]
        cblas_sgemv(
        CblasRowMajor,
        CblasNoTrans,
        L->out_features, L->in_features,
        1.0f,
        L->base.weights,        // A
        L->in_features,         // lda
        input,                  // x
        1,                      // incx
        0.0f,
        L->base.output,         // y
        1                       // incy
        );

        // add biases
        for (size_t o = 0; o < L->out_features; o++) {
            L->base.output[o] += L->biases[o];
        }
    }

    float* linear_backward(void *self, float *gradients, size_t time_step) {
        LinearLayer *L = (LinearLayer*)self;

        // 1) Weight gradients: ΔW += dY * x^T
        // dY: [out_features], x: [in_features]
        cblas_sger(
        CblasRowMajor,
        L->out_features,        // m
        L->in_features,         // n
        1.0f,
        gradients,              // x (m-vector)
        1,                      // incx
        L->base.inputs,         // y (n-vector)
        1,                      // incy
        L->base.weight_gradients, // A (m x n)
        L->in_features          // lda
        );

        // 2) Bias gradients: Δb += dY
        for (size_t o = 0; o < L->out_features; o++) {
            L->base.bias_gradients[o] += gradients[o];
        }

        // 3) Input gradients: dX = W^T * dY
        // Note: W^T: [in_features x out_features], dY: [out_features]
        cblas_sgemv(
        CblasRowMajor,
        CblasTrans,
        L->out_features,        // rows of original W
        L->in_features,         // cols of original W
        1.0f,
        L->base.weights,        // A
        L->in_features,         // lda
        gradients,              // x
        1,                      // incx
        0.0f,
        L->base.input_gradients,// y
        1                       // incy
        );

        return L->base.input_gradients;
    }

    // Update weights for Linear Layer
    void linear_update_weights(void *self, float learning_rate) {
        LinearLayer *layer = (LinearLayer *)self;
        
        // Update weights
        for (size_t i = 0; i < layer->in_features * layer->out_features; i++) {
            layer->base.weights[i] -= learning_rate * layer->base.weight_gradients[i];
        }
        
        // Update biases
        for (size_t i = 0; i < layer->out_features; i++) {
            layer->biases[i] -= learning_rate * layer->base.bias_gradients[i];
        }
    }

    void linear_zero_grad(void *self) {
        LinearLayer *layer = (LinearLayer *)self;
        memset(layer->base.weight_gradients, 0, sizeof(float) * layer->in_features * layer->out_features);
        memset(layer->base.bias_gradients, 0, sizeof(float) * layer->out_features);
        memset(layer->base.input_gradients, 0, sizeof(float) * layer->in_features);
    }

    // Free allocated memory
    void linear_free(LinearLayer *layer) {
        free(layer->base.weights);
        free(layer->biases);
        free(layer->base.weight_gradients);
        free(layer->base.bias_gradients);
        free(layer->base.input_gradients);
        free(layer->output);
    }

