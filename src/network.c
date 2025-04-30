#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "../include/network.h"
#include "../include/utils/nmnist_loader.h"
#include "../include/utils/network_logger.h"
#include "../include/models/lif_neuron.h"
#include "../include/layers/spiking_layer.h"

#define LEARNING_RATE 0.01
#define EPOCHS 10
#define BATCH_SIZE 16

#define ENABLE_DEBUG_LOG


// Create a new network with a given number of layers
Network *create_network(size_t num_layers) {
    Network *network = (Network *)malloc(sizeof(Network));
    network->forward = forward;
    network->layers = (LayerBase **)malloc(num_layers * sizeof(LayerBase *));
    network->num_layers = num_layers;

    // initialize layers 

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

// Perform forward propagation through the network
void forward(Network *network, float *input, size_t input_size) {
    float *current_input = input;
    size_t current_input_size = input_size;

    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        layer->forward(layer, current_input, current_input_size);

        // Update current input to the output of the current layer
        current_input = layer->output; // Correctly reference the layer's output
        current_input_size = layer->output_size; // Update input size for the next layer
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


float calculate_loss(float *output, int label, size_t output_size) {
    // Cross-entropy loss
    float loss = 0.0f;
    for (size_t i = 0; i < output_size; i++) {
        float target = (i == label) ? 1.0f : 0.0f;
        loss += -target * log(output[i]);
    }
    return loss;
}

void update_weights(Network *network, float learning_rate) {
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        if (layer->update_weights) {
            layer->update_weights(layer, learning_rate);
        }
    }
}

// Compute spike probabilities using softmax
void compute_probabilities(float *spike_counts, size_t num_neurons, float *probabilities) {
    float max_value = -INFINITY;
    float sum_exp = 0.0f;

    // Find the maximum spike count for numerical stability
    for (size_t i = 0; i < num_neurons; i++) {
        if (spike_counts[i] > max_value) {
            max_value = spike_counts[i];
        }
    }

    // Compute exponentials and sum them
    for (size_t i = 0; i < num_neurons; i++) {
        probabilities[i] = exp(spike_counts[i] - max_value); // Softmax numerator
        sum_exp += probabilities[i];
    }

    // Normalize to get probabilities
    for (size_t i = 0; i < num_neurons; i++) {
        probabilities[i] /= sum_exp;
    }

   
}

void zero_grads(Network* model) {
    for (size_t i = 0; i < model->num_layers; i++) {
        if(model->layers[i]->zero_grad != NULL) {
            model->layers[i]->zero_grad(model->layers[i]);
        }
    }
}

void train(Network *network, NMNISTDataset *dataset) {
    printf("Starting training...\n");
    //const int TIME_BINS = 311;
    const int TIME_BINS = 8;

    // Arrays to store batch data
    float** batch_probabilities = (float**)malloc(BATCH_SIZE * sizeof(float*));
    int* batch_labels = (int*)malloc(BATCH_SIZE * sizeof(int));
    float** batch_gradients = (float**)malloc(BATCH_SIZE * sizeof(float*));

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float epoch_loss = 0.0;
        size_t total_samples = 0;

        for (size_t batch_start = 0; batch_start < dataset->num_samples; batch_start += BATCH_SIZE) {
            size_t batch_end = batch_start + BATCH_SIZE;
            if (batch_end > dataset->num_samples) batch_end = dataset->num_samples;
            size_t actual_batch_size = batch_end - batch_start;

            zero_grads(network);

            // Process each sample in batch
            for (size_t batch_idx = 0; batch_idx < actual_batch_size; batch_idx++) {
                size_t sample_idx = batch_start + batch_idx;
                NMNISTSample *sample = &dataset->samples[sample_idx];

                // Convert events to input format
                int max_time = sample->events[sample->num_events - 1].timestamp;
                float *input = convert_events_to_input(
                    sample->events, sample->num_events, TIME_BINS, 34, 34, max_time);

                //print_frame(input, 34, 34);

                size_t input_size_per_bin = 34 * 34;

                // Reset spike counts for all layers
                for (size_t l = 0; l < network->num_layers; l++) {
                    if (network->layers[l]->is_spiking == true) {
                        network->layers[l]->reset_spike_counts(network->layers[l]);
                    }
                }

                // Process each time bin sequentially
                for (int t = 0; t < TIME_BINS; t++) {
                    float *frame = &input[t * input_size_per_bin];
                    network->layers[0]->forward(network->layers[0], frame, input_size_per_bin);
                    for (size_t j = 1; j < network->num_layers; j++) {
                        network->layers[j]->forward(network->layers[j], 
                                                   network->layers[j-1]->output,
                                                   network->layers[j-1]->output_size);
                    }

                    #ifdef ENABLE_DEBUG_LOG
                        log_spikes(network, epoch + 1, total_samples + batch_idx, t);
                    #endif
                }

                // Get output spike counts
                SpikingLayer *output_layer = (SpikingLayer *)network->layers[network->num_layers - 1];
                if (output_layer->num_neurons == 0 || output_layer->neurons == NULL) {
                    fprintf(stderr, "Error: Output layer neurons not initialized.\n");
                    free(input);
                    continue;
                }

                float spike_counts[output_layer->num_neurons];
                for (size_t n = 0; n < output_layer->num_neurons; n++) {
                    LIFNeuron *neuron = (LIFNeuron *)output_layer->neurons[n];
                    spike_counts[n] = (float)neuron->spike_count;
                }

                // Store probabilities and labels for batch
                batch_probabilities[batch_idx] = (float*)malloc(output_layer->num_neurons * sizeof(float));
                compute_probabilities(spike_counts, output_layer->num_neurons, batch_probabilities[batch_idx]);
                batch_labels[batch_idx] = sample->label;

                // Calculate loss
                epoch_loss += -log(batch_probabilities[batch_idx][sample->label]);
                total_samples++;

                free(input);
            }

            // Compute average gradients across batch 
            // !!!!! HARDOCDED 10
            float* avg_gradients = (float*)calloc(10, sizeof(float));
            for (size_t batch_idx = 0; batch_idx < actual_batch_size; batch_idx++) {
                // Compute gradients for each sample
                batch_gradients[batch_idx] = (float*)malloc(10 * sizeof(float));
                for (size_t k = 0; k < 10; k++) {
                    
                    // With a surrogate (e.g., sigmoid derivative):
                    float surrogate_grad = 1.0f / (1.0f + fabs(batch_probabilities[batch_idx][k] - (k == batch_labels[batch_idx] ? 1.0f : 0.0f)));
                    batch_gradients[batch_idx][k] = surrogate_grad * (batch_probabilities[batch_idx][k] - (k == batch_labels[batch_idx]  ? 1.0f : 0.0f));
                    avg_gradients[k] += batch_gradients[batch_idx][k];
                }
            }

            // Average the gradients
            for (size_t k = 0; k < 10; k++) {
                avg_gradients[k] /= actual_batch_size;
            }

            // Backward pass with averaged gradients
            float* current_gradients = avg_gradients;

            for (size_t j = network->num_layers; j-- > 0;) {
                current_gradients = network->layers[j]->backward(network->layers[j], current_gradients);
            }

            // Update weights after processing entire batch
            update_weights(network, LEARNING_RATE);

            
        }

        #ifdef ENABLE_DEBUG_LOG
            log_gradients(network, (epoch + 1),  0);
            log_weights(network, epoch + 1, 0);
        #endif

        printf("Epoch %d/%d, Loss: %.4f\n", epoch + 1, EPOCHS, epoch_loss / total_samples);
    }

    // Free batch memory
    for (size_t batch_idx = 0; batch_idx < BATCH_SIZE; batch_idx++) {
        free(batch_probabilities[batch_idx]);
        free(batch_gradients[batch_idx]);
    }
    free(batch_probabilities);
    free(batch_labels);
    free(batch_gradients);
    //free(avg_gradients);

    printf("Training complete!\n");
}


