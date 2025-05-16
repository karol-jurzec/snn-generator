#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>

#include "../include/network.h"
#include "utils/network_logger.h"

#define LEARNING_RATE 0.001
//#define EPOCHS 10
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

void train(Network *network, Dataset *dataset) {
    printf("Starting training...\n");

    int epochs = 50;
    MSECountLoss mse_loss;
    init_mse_count_loss(&mse_loss, BATCH_SIZE, 10, TIME_BINS, 0.8f, 0.2f);

    int* batch_labels = (int*)malloc(BATCH_SIZE * sizeof(int));
    float** batch_gradients = (float**)malloc(BATCH_SIZE * sizeof(float*));

    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0.0f;
        int correct_predictions = 0;
        size_t total_samples = 0;

        for (size_t batch_start = 0; batch_start < dataset->num_samples; batch_start += BATCH_SIZE) {
            size_t batch_end = batch_start + BATCH_SIZE;
            if (batch_end > dataset->num_samples) batch_end = dataset->num_samples;
            size_t actual_batch_size = batch_end - batch_start;

            zero_grads(network);

            for (size_t batch_idx = 0; batch_idx < actual_batch_size; batch_idx++) {
                size_t sample_idx = batch_start + batch_idx;
                Sample *sample = &dataset->samples[sample_idx];

                int max_time = sample->events[sample->num_events - 1].timestamp;
                
                size_t input_size_per_bin = 2 * 34 * 34;
                float *input = (float *)malloc(input_size_per_bin * sizeof(float));
                memcpy(input, sample->input, input_size_per_bin * sizeof(float));

                for (size_t l = 0; l < network->num_layers; l++) {
                    if (network->layers[l]->is_spiking) {
                        network->layers[l]->reset_spike_counts(network->layers[l]);
                    }
                }

                for (int t = 0; t < TIME_BINS; t++) {
                    float *frame = &input[t * input_size_per_bin];
                    network->layers[0]->forward(network->layers[0], frame, input_size_per_bin, 0);
                    for (size_t j = 1; j < network->num_layers; j++) {
                        network->layers[j]->forward(network->layers[j],
                                                    network->layers[j - 1]->output,
                                                    network->layers[j - 1]->output_size, 0);
                    }

                    #ifdef ENABLE_DEBUG_LOG
                        if(epoch == 9) {
                        //    log_spikes(network, epoch + 1, total_samples + batch_idx, t, sample->label);
                        }

                    #endif
                }

                SpikingLayer *output_layer = (SpikingLayer *)network->layers[network->num_layers - 1];
                if (output_layer->num_neurons == 0 || output_layer->neurons == NULL) {
                    fprintf(stderr, "Error: Output layer neurons not initialized.\n");
                    free(input);
                    continue;
                }

                int label = sample->label;
                batch_labels[batch_idx] = label;

                float max_spikes = -1.0f;
                int predicted_label = -1;

                for (size_t n = 0; n < output_layer->num_neurons; n++) {
                    LIFNeuron *neuron = (LIFNeuron *)output_layer->neurons[n];
                    float spike_count = (float)neuron->spike_count;
                    mse_loss.spike_counts[batch_idx * output_layer->num_neurons + n] = spike_count;

                    // Accuracy prediction logic
                    if (spike_count > max_spikes) {
                        max_spikes = spike_count;
                        predicted_label = (int)n;
                    }
                }

                if (predicted_label == label) {
                    correct_predictions++;
                }

                total_samples++;
                free(input);
            }

            generate_target_counts(&mse_loss, batch_labels);
            float batch_loss = compute_mse_loss(&mse_loss);
            epoch_loss += batch_loss;

            int num_classes = NUM_CLASSES;

            // Compute gradients
            float* avg_gradients = (float*)calloc(num_classes, sizeof(float));
            for (size_t batch_idx = 0; batch_idx < actual_batch_size; batch_idx++) {
                for (size_t k = 0; k < num_classes; k++) {
                    float output = mse_loss.spike_counts[batch_idx * num_classes + k];
                    float target = mse_loss.target_counts[batch_idx * num_classes + k];
                    avg_gradients[k] += 2.0f * (output - target);
                }
            }

            for (size_t k = 0; k < 10; k++) {
                avg_gradients[k] /=  (actual_batch_size);
            }

            float* current_gradients = avg_gradients;
            for (size_t j = network->num_layers; j-- > 0;) {
              
                current_gradients = network->layers[j]->backward(network->layers[j], current_gradients, 0);
            }

            update_weights(network, LEARNING_RATE);

            //for (size_t batch_idx = 0; batch_idx < actual_batch_size; batch_idx++) {
            //    free(batch_gradients[batch_idx]);
            //}
            //free(avg_gradients);

            #ifdef ENABLE_DEBUG_LOG
               // log_gradients(network, epoch + 1, epoch * dataset->num_samples + batch_start);
                //log_weights(network, epoch + 1, epoch * dataset->num_samples + batch_start);
            #endif
        }

        float avg_loss = epoch_loss / total_samples;
        float accuracy = 100.0f * ((float)correct_predictions / total_samples);
        printf("Epoch %d/%d, Loss: %.4f, Accuracy: %.2f%%\n",
               epoch + 1, epochs, avg_loss, accuracy);
    }

    
    free_mse_count_loss(&mse_loss);
    free(batch_labels);
    free(batch_gradients);

    printf("Training complete!\n");
}

int is_whitespace_only(const char *str) {
    while (*str) {
        if (!isspace((unsigned char)*str)) return 0;
        str++;
    }
    return 1;
}

float* load_sample_file(const char* filepath) {
    size_t total_values = 84 * 2 * 10 * 10;
    float *input = (float*)malloc(total_values * sizeof(float));
    if (!input) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    FILE *file = fopen(filepath, "r");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", filepath);
        free(input);
        return NULL;
    }

    size_t idx = 0;
    char line[2048];

    while (fgets(line, sizeof(line), file)) {
        // Skip metadata or whitespace-only lines
        if (strstr(line, "Timestep") || strstr(line, "Channel") || is_whitespace_only(line))
            continue;

        // Parse space-separated floats
        char *token = strtok(line, " ");
        while (token && idx < total_values) {
            float value = strtof(token, NULL);
            input[idx++] = value;
            token = strtok(NULL, " ");
        }
    }

    fclose(file);

    if (idx != total_values) {
        fprintf(stderr, "Expected %zu values, but got %zu\n", total_values, idx);
        free(input);
        return NULL;
    }

    return input;
}

void sample_test(Network *network, const char* path) {
    size_t input_size = 2 * 84 * 10 * 10;
    float *input = load_sample_file(path);
    if (!input) {
        fprintf(stderr, "Input loading failed.\n");
        return;
    }

    if (!input) {
        fprintf(stderr, "Input loading failed.\n");
        return;
    }

    // Now run through the network, one frame at a time:
    size_t frame_size = 2 * 10 * 10;

    for (int t = 0; t < 84; ++t) {
        float *frame = &input[t * frame_size];
        network->layers[0]->forward(network->layers[0], frame, frame_size, 0);
        for (size_t l = 1; l < network->num_layers; l++) {
            network->layers[l]->forward(
                network->layers[l], 
                network->layers[l - 1]->output, 
                network->layers[l - 1]->output_size, 0);
        }

       log_spikes(network, 0, 0, t, 0);
       //log_outputs(network, 0, 0, t);
       //log_inputs(network, 0, 0, t);
        
    }

    SpikingLayer *output_layer = (SpikingLayer *)network->layers[network->num_layers - 1];
    float spike_counts[output_layer->num_neurons];
    for (size_t n = 0; n < output_layer->num_neurons; n++) {
        LIFNeuron *neuron = (LIFNeuron *)output_layer->neurons[n];
        spike_counts[n] = (float)neuron->spike_count;
        
    }

    printf("TESTING HAS BEEN FINISHED\n");
}

float test(Network *network, Dataset *dataset) {
    size_t correct_predictions = 0;
    size_t total_samples = dataset->num_samples;

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
            if(i % 1000 == 0) {
                //log_inputs(network, 0, i, t);
                //log_spikes(network, 0, i, t, sample->label);
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

        free(input);
    }

    log_weights(network, 0, 0);

    float accuracy = (float)correct_predictions / total_samples * 100.0f;
    printf("Validation Accuracy: %.2f%%\n", accuracy);
    return accuracy;
}


