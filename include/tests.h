#include "models/model_base.h"
#include "network.h"

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
void print_output(float *output, size_t out_channels, size_t output_dim) ;
void generate_synthetic_input(float *input, size_t size);
void print_output_spikes(float *output, size_t num_neurons);
void nmnist_loader_test();
void discretization_test();
void loader_test();
void train_test();
void test_network_training();
void stmnist_test();

void iris_classification_example(); // fully connected network 
void hor_vert_dataset();


#endif // TESTS_H
