#include "models/model_base.h"

#ifndef TESTS_H
#define TESTS_H

void single_neuron_test(ModelBase *model_base, const char* filename);
void maxpool2d_test();
void network_test();
void spiking_layer_test();
void network_loader_test();
void flatten_test();
void conv2d_test();
void linear_test();
void print_output();
void generate_synthetic_input(float *input, size_t size);
void print_output_spikes(float *output, size_t num_neurons);

#endif // TESTS_H
