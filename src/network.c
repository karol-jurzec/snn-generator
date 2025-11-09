#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <time.h> 
#include <string.h>

#include "../include/network.h"
#include "utils/network_logger.h"
#include "utils/perf.h"

#define BATCH_SIZE 1
#define TIME_BINS 300
#define NUM_CLASSES 10

#define ENABLE_DEBUG_LOG

// Create a new network with a given number of layers
Network *create_network(size_t num_layers) {
    Network *network = (Network *)malloc(sizeof(Network));
    network->forward = forward;
    network->layers = (LayerBase **)malloc(num_layers * sizeof(LayerBase *));
    network->num_layers = num_layers;

    return network;
}

// Add a layer to the network at a specific index
void add_layer(Network *network, LayerBase *layer, size_t index) {
    if (index < network->num_layers) {
        network->layers[index] = layer;
    } else {
        printf("Error: Index out of bounds when adding layer.\n");
    }
}

void forward(Network *network, float *input, size_t input_size, int time_step) {
    float *current_input = input;
    size_t current_input_size = input_size;

    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        
        layer->forward(layer, current_input, current_input_size, time_step);
        
        current_input = layer->output;
        current_input_size = layer->output_size;
    }
}


// Free the allocated memory for the network
void free_network(Network *network) {
    for (size_t i = 0; i < network->num_layers; i++) {
        free(network->layers[i]);
    }
    free(network->layers);
    free(network);
}

// Compute spike probabilities using softmax
void compute_probabilities(float *spike_counts, size_t num_neurons, float *probabilities) {
    float max_value = -INFINITY;
    float sum_exp = 0.0f;

    for (size_t i = 0; i < num_neurons; i++) {
        if (spike_counts[i] > max_value) {
            max_value = spike_counts[i];
        }
    }

    for (size_t i = 0; i < num_neurons; i++) {
        probabilities[i] = exp(spike_counts[i] - max_value); // Softmax numerator
        sum_exp += probabilities[i];
    }

    for (size_t i = 0; i < num_neurons; i++) {
        probabilities[i] /= sum_exp;
    }
}

float test(Network *network, Dataset *dataset) {
	size_t correct_predictions = 0;
	size_t total_samples = dataset->num_samples;

	clock_t start_time = clock();  
	perf_mark_start("inference");

	for (size_t i = 0; i < total_samples; i++) {
		Sample *sample = &dataset->samples[i];

		int max_time = sample->events[sample->num_events - 1].timestamp;
		float *input = sample->input;

		size_t input_size_per_bin = dataset->input_channels * dataset->input_width * dataset->input_height;

		for (size_t l = 0; l < network->num_layers; l++) {
			if (network->layers[l]->is_spiking) {
				network->layers[l]->reset_spike_counts(network->layers[l]);
			}
		}

		for (int t = 0; t < sample->num_bins; t++) {
			float *frame = &input[t * input_size_per_bin];
			network->layers[0]->forward(network->layers[0], frame, input_size_per_bin, 0);
			for (size_t j = 1; j < network->num_layers; j++) {
				network->layers[j]->forward(network->layers[j], network->layers[j - 1]->output,
				                            network->layers[j - 1]->output_size, 0);
			}
		}

		SpikingLayer *output_layer = (SpikingLayer *)network->layers[network->num_layers - 1];
		float spike_counts[output_layer->num_neurons];
		for (size_t n = 0; n < output_layer->num_neurons; n++) {
			LIFNeuron *neuron = (LIFNeuron *)output_layer->neurons[n];
			spike_counts[n] = (float)neuron->spike_count;
		}

		float probabilities[output_layer->num_neurons];
		compute_probabilities(spike_counts, output_layer->num_neurons, probabilities);

		int predicted_label = 0;
		float max_prob = probabilities[0];
		for (size_t p = 1; p < output_layer->num_neurons; p++) {
			if (probabilities[p] > max_prob) {
				max_prob = probabilities[p];
				predicted_label = p;
			}
		}

		if (predicted_label == sample->label) {
			correct_predictions++;
		}

		//free(input);
	}

	clock_t end_time = clock(); 

	double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
	double avg_time_per_sample = elapsed_time / total_samples;

	float accuracy = (float)correct_predictions / total_samples * 100.0f;
	printf("Validation Accuracy: %.2f%%\n", accuracy);
	printf("Average Inference Time per Sample: %.6f seconds\n", avg_time_per_sample);

	perf_mark_end("inference");
	perf_add_metric("inference_avg_sec_per_sample", avg_time_per_sample);
	perf_add_metric("inference_accuracy_percent", (double)accuracy);

	return accuracy;
}

int  predict_single_sample(Network *network, Sample *sample, Dataset *dataset) {
    float *input = sample->input;
    size_t input_size_per_bin = dataset->input_channels * dataset->input_width * dataset->input_height;

    for (size_t l = 0; l < network->num_layers; l++) {
        if (network->layers[l]->is_spiking) {
            network->layers[l]->reset_spike_counts(network->layers[l]);
        }
    }

    for (int t = 0; t < sample->num_bins; t++) {
        float *frame = &input[t * input_size_per_bin];
        network->layers[0]->forward(network->layers[0], frame, input_size_per_bin, 0);
        for (size_t j = 1; j < network->num_layers; j++) {
            network->layers[j]->forward(network->layers[j], network->layers[j - 1]->output,
                                        network->layers[j - 1]->output_size, 0);
        }
    }

    SpikingLayer *output_layer = (SpikingLayer *)network->layers[network->num_layers - 1];
    float spike_counts[output_layer->num_neurons];
    for (size_t n = 0; n < output_layer->num_neurons; n++) {
        LIFNeuron *neuron = (LIFNeuron *)output_layer->neurons[n];
        spike_counts[n] = (float)neuron->spike_count;
    }

    float probabilities[output_layer->num_neurons];
    compute_probabilities(spike_counts, output_layer->num_neurons, probabilities);

    int predicted_label = 0;
    float max_prob = probabilities[0];
    for (size_t p = 1; p < output_layer->num_neurons; p++) {
        if (probabilities[p] > max_prob) {
            max_prob = probabilities[p];
            predicted_label = p;
        }
    }

    return predicted_label;
}

void optimize_network(Network* network) {
    (void)network;
}


