#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "../include/network.h"
#include "../include/utils/nmnist_loader.h"
#include "../include/models/lif_neuron.h"
#include "../include/layers/spiking_layer.h"

#define LEARNING_RATE 0.01
#define EPOCHS 10
#define BATCH_SIZE 16


// Create a new network with a given number of layers
Network *create_network(size_t num_layers) {
    Network *network = (Network *)malloc(sizeof(Network));
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
        
        for (size_t i = 0; i < layer->output_size && i < 10; i++) {
            printf("output[%zu] = %f\n", i, layer->output[i]);
        }
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

void train(Network *network, NMNISTDataset *dataset) {
    printf("Starting training...\n");

    const int TIME_BINS = 16; // Example time bins for spike counting

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float epoch_loss = 0.0f;
        size_t total_samples = 0;

        for (size_t batch_start = 0; batch_start < dataset->num_samples; batch_start += BATCH_SIZE) {
            size_t batch_end = batch_start + BATCH_SIZE;
            if (batch_end > dataset->num_samples) batch_end = dataset->num_samples;

            // Batch processing
            for (size_t i = batch_start; i < batch_end; i++) {
                NMNISTSample *sample = &dataset->samples[i];

                // Convert events to input format
                float *input = convert_events_to_input(
                    sample->events, sample->num_events, TIME_BINS, 28, 28, 100000);

                //visualize_sample_frames(&dataset->samples[i], "out/sample__frames", 16, 28, 28, 100000);
                size_t input_size = TIME_BINS * 28 * 28;

                // Forward pass
                network->layers[0]->forward(network->layers[0], input, input_size);
                for (size_t j = 1; j < network->num_layers; j++) {
                    network->layers[j]->forward(network->layers[j], network->layers[j - 1]->output,
                                                network->layers[j - 1]->output_size);
                }

                // Retrieve the last layer and validate it as a SpikingLayer
                LayerBase *last_layer = network->layers[network->num_layers - 1];
                if (last_layer == NULL || last_layer->forward == NULL) {
                    fprintf(stderr, "Error: Output layer is NULL or not properly initialized.\n");
                    free(input);
                    continue;
                }

                SpikingLayer *output_layer = (SpikingLayer *)last_layer;
                if (output_layer->num_neurons == 0 || output_layer->neurons == NULL) {
                    fprintf(stderr, "Error: Output layer neurons are not properly initialized.\n");
                    free(input);
                    continue;
                }

                // Retrieve spike counts from output layer neurons
                float spike_counts[output_layer->num_neurons];
                for (size_t n = 0; n < output_layer->num_neurons; n++) {
                    LIFNeuron *neuron = (LIFNeuron *)output_layer->neurons[n];
                    spike_counts[n] = (float)neuron->spike_count; // Get spike count
                }

                // Compute probabilities
                float probabilities[output_layer->num_neurons];
                compute_probabilities(spike_counts, output_layer->num_neurons, probabilities);

                for (size_t n = 0; n < output_layer->num_neurons; n++) {
                    printf("Neuron[%zu] Spike Count: %f, Probability: %f\n", n, spike_counts[n], probabilities[n]);
                }


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
                    if (layer->backward) {
                        // DebugV: Check output size before backward
                        //printf("Backward pass for layer %zu: output_size = %d\n", j, layer->output_size);

                        layer->backward(layer, gradients);

                        // Resize gradients for the next layer
                        size_t next_grad_size = layer->input_gradients
                                                    ? layer->output_size
                                                    : output_layer->num_neurons;
                        float *new_gradients = (float *)malloc(next_grad_size * sizeof(float));
                        if (new_gradients) {
                            if (layer->input_gradients) {
                                memcpy(new_gradients, layer->input_gradients, next_grad_size * sizeof(float));
                            } else {
                                memset(new_gradients, 0, next_grad_size * sizeof(float));
                                fprintf(stderr, "Warning: layer->input_gradients is NULL, initializing new_gradients to zero.\n");
                            }

                            free(gradients);
                            gradients = new_gradients;
                        } else {
                            fprintf(stderr, "Error: Memory allocation failed for new_gradients.\n");
                            free(gradients);
                            break;
                        }
                    }
                }

                free(gradients); // Free gradients array after backward pass

                // Reset spike counts for the next sample
                for (size_t n = 0; n < output_layer->num_neurons; n++) {
                    LIFNeuron *neuron = (LIFNeuron *)output_layer->neurons[n];
                    neuron->spike_count = 0;
                }

                free(input); // Free the converted input
                total_samples++;
            }       

            // Update weights after processing the batch
            update_weights(network, LEARNING_RATE);
        }

        // Print epoch summary
        printf("Epoch %d/%d, Loss: %.4f\n", epoch + 1, EPOCHS, epoch_loss / total_samples);
    }

    printf("Training complete!\n");
}
