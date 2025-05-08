#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "../include/network.h"
#include "../include/layers/maxpool2d_layer.h"
#include "../include/utils/nmnist_loader.h"
#include "../include/utils/network_logger.h"
#include "../include/models/lif_neuron.h"
#include "../include/layers/spiking_layer.h"
#include "../include/utils/mse_count_loss.h"

//#define LEARNING_RATE 0.01
#define EPOCHS 10
#define BATCH_SIZE 8
#define TIME_BINS 32

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

// Network-wide functions for BPTT
// TODO: change shape of input to [T][B][H * W] 
// T - nubmer of time steps
// B - number of batches 
// H * W - size of single time_step input (in this case H is height and W is widht)
// TODO: process forward and backward through layers in batches


void forward(Network *network, float *input, size_t input_size, int time_step) {
    float *current_input = input;
    size_t current_input_size = input_size;

    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        
        // Store the previous output if needed
        // TODO: check why this makes a segmentation fault 

        if (time_step > 0 && layer->output_history != NULL) {
            memcpy(&layer->output_history[(time_step-1)*layer->output_size], 
                   layer->output, 
                   layer->output_size * sizeof(float));
        }
        
        layer->forward(layer, current_input, current_input_size, time_step);
        
        current_input = layer->output;
        current_input_size = layer->output_size;
    }
}

float* backward(Network *network, float *gradients, int time_step) {
    float *current_gradients = gradients;
    
    for (size_t j = network->num_layers; j-- > 0;) {
        LayerBase *layer = network->layers[j];

        float mx_value = 0.0f; 
        for(int i = 0; i < layer->output_size; ++i) {
            if(fabs(current_gradients[i]) > mx_value) {
                mx_value = fabs(current_gradients[i]);
            }
        }

        printf("Max value for layer %d: %f\n", j, mx_value);
        
        // If this layer has temporal states, load them
        if (layer->output_history != NULL) {
            memcpy(layer->output, 
                   &layer->output_history[time_step * layer->output_size],
                   layer->output_size * sizeof(float));
        }
        
        current_gradients = layer->backward(layer, current_gradients, time_step);
    }
    
    return current_gradients;
}

float compute_loss(Network *network, MSECountLoss *loss_fn, int *labels, int batch_size) {
    SpikingLayer *output_layer = (SpikingLayer *)network->layers[network->num_layers-1];
    
    // Collect spike counts for all samples in batch
    for (int b = 0; b < batch_size; b++) {
        for (size_t n = 0; n < output_layer->num_neurons; n++) {
            LIFNeuron *neuron = (LIFNeuron *)output_layer->neurons[n];
            loss_fn->spike_counts[b * output_layer->num_neurons + n] = (float)neuron->spike_count;
        }
    }
    
    generate_target_counts(loss_fn, labels);
    return compute_mse_loss(loss_fn);
}

void update_weights(Network *network, float learning_rate) {
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        if (layer->update_weights) {
            layer->update_weights(layer, learning_rate);
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

// Reset all layer states between batches
void reset_layer_states(Network *network) {
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        if (layer->is_spiking) {
            SpikingLayer *slayer = (SpikingLayer *)layer;
            for (size_t n = 0; n < slayer->num_neurons; n++) {
                LIFNeuron *neuron = (LIFNeuron *)slayer->neurons[n];
                neuron->spike_count = 0;
                neuron->base.v = 0.0f;
            }
        }
        
        // Clear temporal histories if they exist
        if (layer->output_history) {
            memset(layer->output_history, 0, layer->output_size * TIME_BINS * sizeof(float));
        }
    }
}

// Initialize network with temporal buffers
void initialize_network_with_bptt(Network *network, int time_steps) {
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        layer->time_steps = time_steps;
        
        // Allocate history buffers based on layer type
        if (layer->is_spiking) {
            SpikingLayer *slayer = (SpikingLayer *)layer;
            slayer->membrane_history = (float*)malloc(time_steps * slayer->num_neurons * sizeof(float));
            slayer->spike_history = (int*)malloc(time_steps * slayer->num_neurons * sizeof(int));
        }
        
        // Allocate output history for all layers
        layer->output_history = (float*)malloc(time_steps * layer->output_size * sizeof(float));
        
        // Special case for MaxPool
        if (layer->layer_type == LAYER_MAXPOOL2D) {
            MaxPool2DLayer *mpool = (MaxPool2DLayer *)layer;
            mpool->max_indices_history = (size_t*)malloc(time_steps * mpool->base.output_size * sizeof(size_t));
        }
    }
}

void train(Network *network, NMNISTDataset *dataset) {
    printf("Starting training with BPTT...\n");
    const float LEARNING_RATE = 0.02f;  // Matching PyTorch's 2e-2

    // Initialize loss function (matches PyTorch's parameters)
    MSECountLoss mse_loss;
    init_mse_count_loss(&mse_loss, BATCH_SIZE, 10, TIME_BINS, 0.8f, 0.2f);
    initialize_network_with_bptt(network, TIME_BINS);

    // Batch buffers
    int* batch_labels = (int*)malloc(BATCH_SIZE * sizeof(int));
    float** batch_inputs = (float**)malloc(BATCH_SIZE * sizeof(float*));
    
    // For tracking performance
    float epoch_loss = 0.0f;
    int correct_predictions = 0;
    size_t total_samples = 0;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        epoch_loss = 0.0f;
        correct_predictions = 0;
        total_samples = 0;

        for (size_t batch_start = 0; batch_start < dataset->num_samples; batch_start += BATCH_SIZE) {
            size_t batch_end = batch_start + BATCH_SIZE;
            if (batch_end > dataset->num_samples) batch_end = dataset->num_samples;
            size_t actual_batch_size = batch_end - batch_start;

            // Reset gradients and temporal states
            zero_grads(network);
            reset_layer_states(network);

            // --- PHASE 1: Forward pass (all time steps) ---
            for (int batch_idx = 0; batch_idx < actual_batch_size; batch_idx++) {
                size_t sample_idx = batch_start + batch_idx;
                NMNISTSample *sample = &dataset->samples[sample_idx];
                
                // Convert events to input frames (all time steps)
                int max_time = sample->events[sample->num_events - 1].timestamp;
                batch_inputs[batch_idx] = convert_events_to_input(sample->events, sample->num_events,
                                                                TIME_BINS, 34, 34, max_time);
                batch_labels[batch_idx] = sample->label;
            }

            // Process all time steps for the entire batch
            for (int t = 0; t < TIME_BINS; t++) {
                for (int batch_idx = 0; batch_idx < actual_batch_size; batch_idx++) {
                    // In your train function, before forward pass:
                    float *frame = &batch_inputs[batch_idx][t * 34 * 34 * 2];

                    forward(network, frame, 34*34*2, t);

                    #ifdef ENABLE_DEBUG_LOG
                    if (epoch == EPOCHS-1) {
                        log_spikes(network, epoch+1, batch_start+batch_idx, t, batch_labels[batch_idx]);
                    }
                    #endif
                }
            }

            // --- PHASE 2: Compute loss ---
            SpikingLayer *output_layer = (SpikingLayer *)network->layers[network->num_layers-1];
            float batch_loss = compute_loss(network, &mse_loss, batch_labels, actual_batch_size);
            epoch_loss += batch_loss;

            // Calculate accuracy
            for (int batch_idx = 0; batch_idx < actual_batch_size; batch_idx++) {
                int predicted_label = -1;
                float max_spikes = -1.0f;
                
                for (size_t n = 0; n < output_layer->num_neurons; n++) {
                    LIFNeuron *neuron = (LIFNeuron *)output_layer->neurons[n];
                    if ((float)neuron->spike_count > max_spikes) {
                        max_spikes = (float)neuron->spike_count;
                        predicted_label = (int)n;
                    }
                }
                
                if (predicted_label == batch_labels[batch_idx]) {
                    correct_predictions++;
                }
                total_samples++;
            }

            // --- PHASE 3: Backward pass (reverse time) ---
            float* output_gradients = (float*)calloc(output_layer->num_neurons, sizeof(float));
            
            // Get initial gradients from loss function
            for (int batch_idx = 0; batch_idx < actual_batch_size; batch_idx++) {
                for (size_t n = 0; n < output_layer->num_neurons; n++) {
                    float output = mse_loss.spike_counts[batch_idx * output_layer->num_neurons + n];
                    float target = mse_loss.target_counts[batch_idx * output_layer->num_neurons + n];
                    output_gradients[n] += 2.0f * (output - target) / 
                                         (actual_batch_size * output_layer->num_neurons * TIME_BINS);
                }
            }

            for(int i = 0; i < 10; ++i) {
                float val = output_gradients[i];
                int v = 0;
            }

            // Backpropagate through time
            for (int t = TIME_BINS-1; t >= 0; t--) {
                for (int batch_idx = 0; batch_idx < actual_batch_size; batch_idx++) {
                    backward(network, output_gradients, t);
                }
            }
            free(output_gradients);

            // --- PHASE 4: Update weights ---
            update_weights(network, LEARNING_RATE);

            // Free batch inputs
            for (int batch_idx = 0; batch_idx < actual_batch_size; batch_idx++) {
                free(batch_inputs[batch_idx]);
            }

            printf("Epoch %d, Batch %d/%d, Loss: %.4f, Accuracy: %.2f%%\n",
                  epoch+1, (int)(batch_start/BATCH_SIZE)+1, 
                  (int)(dataset->num_samples/BATCH_SIZE),
                  batch_loss/actual_batch_size,
                  100.0f * correct_predictions / total_samples);
            
            log_gradients(network, epoch, batch_start);
            log_weights(network, epoch, batch_start);
        }

        float avg_loss = epoch_loss / total_samples;
        float accuracy = 100.0f * ((float)correct_predictions / total_samples);
        printf("Epoch %d/%d, Avg Loss: %.4f, Accuracy: %.2f%%\n",
              epoch+1, EPOCHS, avg_loss, accuracy);
    }

    // Cleanup
    free_mse_count_loss(&mse_loss);
    free(batch_labels);
    free(batch_inputs);
    printf("Training complete!\n");
}


