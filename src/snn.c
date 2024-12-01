#include "../layers/lif_layer.c"
#include "../layers/conv2d_layer.c"
#include "../layers/max_pool2d_layer.c"
#include "../layers/linear_layer.c"

typedef struct {
    Conv2DLayer conv1;       
    MaxPool2DLayer maxpool1; 
    Conv2DLayer conv2;       
    MaxPool2DLayer maxpool2; 
    LinearLayer fc;          
    LIFLayer output_layer;   
} SNN;

#include <stdlib.h>

float *flatten_forward(float **input, int rows, int cols) {
    float *output = (float *)malloc(rows * cols * sizeof(float));
    int idx = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[idx++] = input[i][j];
        }
    }
    return output;
}

void initialize_snn(SNN *network) {
    initialize_conv2d_layer(&network->conv1, 5, 1, 0);
    initialize_maxpool2d_layer(&network->maxpool1, 2, 2);
    initialize_conv2d_layer(&network->conv2, 5, 1, 0);
    initialize_maxpool2d_layer(&network->maxpool2, 2, 2);
    initialize_linear_layer(&network->fc, 800, 10);
    initalize_layer(&network->output_layer, 0.0, 0.1, 0.0, 0.5);
}

void update_snn(SNN *network, float **image_input, int image_size) {
    float **conv1_output = conv2d_forward(&network->conv1, image_input, image_size);
    int conv1_output_size = (image_size - network->conv1.kernel_size) + 1;

    float **maxpool1_output = maxpool2d_forward(&network->maxpool1, conv1_output, conv1_output_size);
    int maxpool1_output_size = conv1_output_size / 2;

    float **conv2_output = conv2d_forward(&network->conv2, maxpool1_output, maxpool1_output_size);
    int conv2_output_size = (maxpool1_output_size - network->conv2.kernel_size) + 1;

    float **maxpool2_output = maxpool2d_forward(&network->maxpool2, conv2_output, conv2_output_size);

    int flattened_size = conv2_output_size / 2 * conv2_output_size / 2;
    float *flattened_output = flatten_forward(maxpool2_output, conv2_output_size / 2, conv2_output_size / 2);
    float *fc_output = linear_forward(&network->fc, flattened_output);

    float output_input_currents[NUM_NEURONS] = {0};
    for (int i = 0; i < NUM_NEURONS && i < network->fc.out_features; i++) {
        output_input_currents[i] = fc_output[i];
    }

    update_layer(&network->output_layer, output_input_currents);


    for (int i = 0; i < conv1_output_size; i++) free(conv1_output[i]);
    free(conv1_output);
    for (int i = 0; i < maxpool1_output_size; i++) free(maxpool1_output[i]);
    free(maxpool1_output);
    for (int i = 0; i < conv2_output_size; i++) free(conv2_output[i]);
    free(conv2_output);
    for (int i = 0; i < conv2_output_size / 2; i++) free(maxpool2_output[i]);
    free(maxpool2_output);
    free(flattened_output);
    free(fc_output);
}

int get_snn_output(SNN *network) {
    int output_class = -1;
    int max_spikes = -1;

    for (int i = 0; i < NUM_NEURONS; ++i) {
        int spike_count = network->output_layer.neurons[i].spike_count;
        if (spike_count > max_spikes) {
            output_class = i;
            max_spikes = spike_count;
        }
    }

    return output_class;
}

void get_snn_spike_counts(SNN *network) {
    for (int i = 0; i < NUM_NEURONS; ++i) {
        int spike_count = network->output_layer.neurons[i].spike_count;
        printf("Class %d: %d spikes\n", i, spike_count);
    }
}
