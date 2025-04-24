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

    const int TIME_BINS = 311; 

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float epoch_loss = 0.0;
        size_t total_samples = 0;

        //log_gradients(network, 0, 0);
        //log_weights(network, 0, 0);

        for (size_t batch_start = 0; batch_start < dataset->num_samples; batch_start += BATCH_SIZE) {
            size_t batch_end = batch_start + BATCH_SIZE;
            if (batch_end > dataset->num_samples) batch_end = dataset->num_samples;

            zero_grads(network);

            // Batch processing
            for (size_t i = batch_start; i < batch_end; i++) {
                NMNISTSample *sample = &dataset->samples[i];

                // Convert events to input format
                int max_time = sample->events[sample->num_events - 1].timestamp;
                float *input = convert_events_to_input(
                    sample->events, sample->num_events, TIME_BINS, 34, 34, max_time);

                size_t input_size_per_bin = 34 * 34; //to do -- probably one channel

                for (size_t l = 0; l < network->num_layers; l++) {
                    if (network->layers[l]->reset_spike_counts) {
                        network->layers[l]->reset_spike_counts(network->layers[l]);
                    }
                }

                // Process each time bin sequentially
                for (int t = 0; t < TIME_BINS; t++) {
                    float *frame = &input[t * input_size_per_bin];

                    // Forward pass for the current time bin
                    network->layers[0]->forward(network->layers[0], frame, input_size_per_bin);
                    for (size_t j = 1; j < network->num_layers; j++) {
                        network->layers[j]->forward(network->layers[j], network->layers[j - 1]->output,
                                                    network->layers[j - 1]->output_size);
                    }

                    //log_inputs(network, epoch + 1, total_samples, t);
                    log_spikes(network, epoch + 1, total_samples, t);
                    //log_membranes(network, epoch + 1, total_samples, t);
                }

                // Retrieve the last layer and validate it as a SpikingLayer
                SpikingLayer *output_layer = (SpikingLayer *)network->layers[network->num_layers - 1];
                if (output_layer->num_neurons == 0 || output_layer->neurons == NULL) {
                    fprintf(stderr, "Error: Output layer neurons are not properly initialized.\n");
                    free(input);
                    continue;
                }

                // Retrieve spike counts from output layer neurons
                float spike_counts[output_layer->num_neurons];
                for (size_t n = 0; n < output_layer->num_neurons; n++) {
                    LIFNeuron *neuron = (LIFNeuron *)output_layer->neurons[n];
                    spike_counts[n] = (float)neuron->spike_count; // Get accumulated spike count
                }

                // Compute probabilities

                float probabilities[output_layer->num_neurons];
                compute_probabilities(spike_counts, output_layer->num_neurons, probabilities);

                // Calculate loss (Cross-Entropy Loss)
                int label = sample->label;
                float loss = -log(probabilities[label]); // Cross-Entropy for one-hot target
                epoch_loss += loss;

                // Backward pass
                float *gradients = (float *)malloc(output_layer->num_neurons * sizeof(float));
                for (size_t k = 0; k < output_layer->num_neurons; k++) {
                    gradients[k] = probabilities[k] - (k == label ? 1.0f : 0.0f); // Gradient of Cross-Entropy
                }

                for (size_t j = network->num_layers; j-- > 0;) {
                    LayerBase *layer = network->layers[j];
                    gradients = layer->backward(layer, gradients);

                }

                log_gradients(network, 0, total_samples);

                //free(gradients); // Free gradients array after backward pass
                free(input);     // Free the converted input
                total_samples++;
            }

            #ifdef ENABLE_DEBUG_LOG

            log_weights(network, epoch + 1, batch_start);
            //log_gradients(network, epoch + 1, batch_start);

            #endif

            // Update weights after processing the batch
            update_weights(network, LEARNING_RATE);

        }

        // Print epoch summary
        printf("Epoch %d/%d, Loss: %.4f\n", epoch + 1, EPOCHS, epoch_loss / total_samples);
        

        /* logging weights and biasess to out/ directory
            for insight into data 
        */

        #ifdef ENABLE_DEBUG_LOG

        //log_weights(network, epoch + 1);
        //log_gradients(network, epoch + 1);

        #endif
    }

    printf("Training complete!\n");
}


