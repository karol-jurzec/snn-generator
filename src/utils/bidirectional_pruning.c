#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../../include/utils/bidirectional_pruning.h"
#include "../../include/utils/pruning_utils.h"
#include "../../include/utils/network_loader.h"
#include "../../include/utils/perf.h"
#include "../../include/models/lif_neuron.h"
#include "../../include/layers/layer_base.h"

BidirectionalPruningInfo* create_bidirectional_info(Network *network) {
    BidirectionalPruningInfo *info = (BidirectionalPruningInfo *)malloc(sizeof(BidirectionalPruningInfo));
    
    info->num_layers = network->num_layers;
    info->inactive_neurons_count = (int *)calloc(network->num_layers, sizeof(int));
    info->layer_activity_ratio = (float *)calloc(network->num_layers, sizeof(float));
    
    // Backward pruning arrays
    info->inactive_out_channels = (bool **)malloc(network->num_layers * sizeof(bool *));
    info->pruned_out_channels = (size_t *)calloc(network->num_layers, sizeof(size_t));
    
    // Forward pruning arrays  
    info->inactive_in_channels = (bool **)malloc(network->num_layers * sizeof(bool *));
    info->pruned_in_channels = (size_t *)calloc(network->num_layers, sizeof(size_t));
    
    // Connection mapping (including MaxPool2d support)
    info->backward_connections = (int *)malloc(network->num_layers * sizeof(int));
    info->forward_connections = (int *)malloc(network->num_layers * sizeof(int));
    
    // MaxPool2d tracking arrays 
    info->maxpool_between_backward = (int *)malloc(network->num_layers * sizeof(int));
    info->maxpool_between_forward = (int *)malloc(network->num_layers * sizeof(int));
    
    // Initialize all connections to -1 (no connection/no MaxPool)
    for (size_t i = 0; i < network->num_layers; i++) {
        info->backward_connections[i] = -1;
        info->forward_connections[i] = -1;
        info->maxpool_between_backward[i] = -1;
        info->maxpool_between_forward[i] = -1;
        info->inactive_out_channels[i] = NULL;
        info->inactive_in_channels[i] = NULL;
    }
    
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        if (layer->layer_type == LAYER_CONV2D) {
            Conv2DLayer *conv = (Conv2DLayer *)layer;
            info->inactive_out_channels[i] = (bool *)calloc(conv->out_channels, sizeof(bool));
            info->inactive_in_channels[i] = (bool *)calloc(conv->in_channels, sizeof(bool));
        }
    }
    
    // Initialize statistics
    info->total_backward_reduction = 0.0f;
    info->total_forward_reduction = 0.0f;
    info->combined_reduction = 0.0f;
    
    return info;
}

// Maps layer connections including MaxPool2d support for accurate channel-to-neuron mapping
void map_layer_connections(Network *network, BidirectionalPruningInfo *info) {
    printf("\n=== mapping (including MaxPool2d) ===\n");
    
    // Simplified loop - skip last 2 layers since we look ahead
    for (size_t i = 0; i < network->num_layers - 2; i++) {
        LayerBase *current = network->layers[i];
        
        // BACKWARD: Conv2d → [MaxPool2d] → Spiking
        if (current->layer_type == LAYER_CONV2D) {
            LayerBase *successor = network->layers[i + 1];
            LayerBase *second_successor = network->layers[i + 2];
            
            // Case 1: Conv2d → MaxPool2d → Spiking
            if (successor->layer_type == LAYER_MAXPOOL2D && second_successor->is_spiking) {
                info->backward_connections[i] = i + 2;
                info->maxpool_between_backward[i] = i + 1;
                printf("Backward: Conv2D[%zu] → MaxPool2D[%zu] → Spiking[%zu]\n", i, i+1, i+2);
            }
            // Case 2: Conv2d → Spiking (direct)
            else if (successor->is_spiking) {
                info->backward_connections[i] = i + 1;
                printf("Backward: Conv2D[%zu] → Spiking[%zu]\n", i, i+1);
            }
        }
        
        if (current->is_spiking) {
            for (size_t j = i + 1; j < network->num_layers - 1; j++) {
                if (network->layers[j]->layer_type == LAYER_CONV2D) {
                    info->forward_connections[i] = j;
                    printf("Forward: Spiking[%zu] → Conv2D[%zu]\n", i, j);
                    break;
                }
            }
        }
    }
}

// Analyzes Conv2D → [MaxPool2D] → Spiking connections for backward channel pruning
void analyze_backward_connections(Network *network, BidirectionalPruningInfo *info, int threshold) {
    printf("\n--- BACKWARD PRUNING: Conv2D → [MaxPool2D] → Spiking Analysis ---\n");
    
    for (size_t i = 0; i < network->num_layers; i++) {
        if (info->backward_connections[i] != -1) {
            size_t spiking_idx = info->backward_connections[i];
            size_t maxpool_idx = info->maxpool_between_backward[i];
            
            Conv2DLayer *conv = (Conv2DLayer *)network->layers[i];
            SpikingLayer *spiking = (SpikingLayer *)network->layers[spiking_idx];
            
            printf("Analyzing Conv2D[%zu] → Spiking[%zu]\n", i, spiking_idx);
            printf("Conv2D: %d→%d channels, %dx%d input\n", 
                   conv->in_channels, conv->out_channels, conv->input_dim, conv->input_dim);
            printf("Spiking: %zu neurons\n", spiking->num_neurons);
            
            int out_h = (conv->input_dim + 2*conv->padding - conv->kernel_size) / conv->stride + 1;
            int out_w = out_h;  // assuming square input
            
            // Apply MaxPool2d if present
            if (maxpool_idx != -1) {
                MaxPool2DLayer *maxpool = (MaxPool2DLayer *)network->layers[maxpool_idx];
                out_h = out_h / maxpool->kernel_size;
                out_w = out_w / maxpool->kernel_size;
                
                printf("Conv2D → MaxPool2D[%d] (%dx%d) → Final: %dx%d\n", 
                       maxpool_idx, maxpool->kernel_size, maxpool->kernel_size, out_h, out_w);
            } else {
                printf("Conv2D output (no pooling): %dx%d\n", out_h, out_w);
            }

            int neurons_per_channel = out_h * out_w;
            printf("Mapping: %dx%d=%d neurons per channel\n", out_h, out_w, neurons_per_channel);
            
            for (int channel = 0; channel < conv->out_channels; channel++) {
                bool channel_inactive = true;
                int active_neurons_in_channel = 0;
                
                for (int h = 0; h < out_h; h++) {
                    for (int w = 0; w < out_w; w++) {
                        int neuron_idx = channel * neurons_per_channel + h * out_w + w;
                        
                        if (neuron_idx < spiking->num_neurons) {
                            if (spiking->total_spikes[neuron_idx] > threshold) {
                                channel_inactive = false;
                                active_neurons_in_channel++;
                            }
                        }
                    }
                }
                
                if (channel_inactive) {
                    info->inactive_out_channels[i][channel] = true;
                    info->pruned_out_channels[i]++;
                    printf("Channel %d: inactive (%d spikes total)\n", channel, 0);
                } else {
                    printf("Channel %d: active (%d neurons spiking)\n", channel, active_neurons_in_channel);
                }
            }
            
            float pruning_ratio = (float)info->pruned_out_channels[i] / conv->out_channels * 100.0f;
        }
    }
}

void analyze_forward_connections(Network *network, BidirectionalPruningInfo *info) {    
    for (size_t i = 0; i < network->num_layers; i++) {
        if (info->forward_connections[i] != -1) {
            size_t conv_idx = info->forward_connections[i];
            
            LayerBase *spiking_layer = network->layers[i];
            LayerBase *conv_layer = network->layers[conv_idx];
            
            SpikingLayer *spiking = (SpikingLayer *)spiking_layer;
            Conv2DLayer *conv = (Conv2DLayer *)conv_layer;
                        
            Conv2DLayer *prev_conv = NULL;
            for (int j = i - 1; j >= 0; j--) {
                if (network->layers[j]->layer_type == LAYER_CONV2D) {
                    prev_conv = (Conv2DLayer *)network->layers[j];
                    printf("   Input pochodzi z Conv2D[%d]: %d channels\n", j, prev_conv->out_channels);
                    
                    if (info->inactive_out_channels[j]) {
                        size_t propagated_count = 0;
                        
                        for (int ch = 0; ch < prev_conv->out_channels && ch < conv->in_channels; ch++) {
                            if (info->inactive_out_channels[j][ch]) {
                                info->inactive_in_channels[conv_idx][ch] = true;
                                propagated_count++;
                            }
                        }
                        
                        info->pruned_in_channels[conv_idx] = propagated_count;
                        
                        printf("Propagated %zu inactive channels to Conv2D[%zu] input\n", 
                               propagated_count, conv_idx);
                    } else {
                        printf("No inactive channels to propagate from Conv2D[%d]\n", j);
                    }
                    break;
                }
            }
            
        }
    }
}

BidirectionalPruningInfo* analyze_bidirectional_activity(Network *network, int threshold) {
    printf("\n=== BIDIRECTIONAL PRUNING ANALYSIS (MaxPool2d aware) ===\n");
    printf("Threshold: %d spikes\n", threshold);
    
    BidirectionalPruningInfo *info = create_bidirectional_info(network);
    
    map_layer_connections(network, info);
    analyze_backward_connections(network, info, threshold);
    analyze_forward_connections(network, info);
    calculate_pruning_statistics(network, info);
    
    return info;
}

void calculate_pruning_statistics(Network *network, BidirectionalPruningInfo *info) {
    size_t total_out_channels = 0, total_pruned_out = 0;
    size_t total_in_channels = 0, total_pruned_in = 0;
    
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        
        if (layer->layer_type == LAYER_CONV2D) {
            Conv2DLayer *conv = (Conv2DLayer *)layer;
            
            // Backward pruning stats
            total_out_channels += conv->out_channels;
            total_pruned_out += info->pruned_out_channels[i];
            
            // Forward pruning stats
            total_in_channels += conv->in_channels;
            total_pruned_in += info->pruned_in_channels[i];
        }
    }
    
    info->total_backward_reduction = (float)total_pruned_out / total_out_channels * 100.0f;
    info->total_forward_reduction = (float)total_pruned_in / total_in_channels * 100.0f;
    
    float backward_efficiency = 1.0f - info->total_backward_reduction / 100.0f;
    float forward_efficiency = 1.0f - info->total_forward_reduction / 100.0f;
    info->combined_reduction = (1.0f - backward_efficiency * forward_efficiency) * 100.0f;
    
    printf("\nPRUNING STATISTICS:\n");
    printf("Backward reduction: %.1f%% (%zu/%zu out_channels)\n", 
           info->total_backward_reduction, total_pruned_out, total_out_channels);
    printf("Forward reduction:  %.1f%% (%zu/%zu in_channels)\n", 
           info->total_forward_reduction, total_pruned_in, total_in_channels);
    printf("Combined reduction: ~%.1f%% (theoretical)\n", info->combined_reduction);
}

void reset_bidirectional_pruning(Network *network) {    
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        
        if (layer->layer_type == LAYER_CONV2D) {
            Conv2DLayer *conv = (Conv2DLayer *)layer;
            
            if (conv->out_active_channels_idx) {
                free(conv->out_active_channels_idx);
                conv->out_active_channels_idx = NULL;
            }
            
            if (conv->in_active_channels_idx) {
                free(conv->in_active_channels_idx);
                conv->in_active_channels_idx = NULL;
            }
            
            conv->out_channels = conv->original_out_channels;
            conv->in_channels = conv->original_in_channels;
            
        }
    }
    
    // Reset spike counters
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        
        if (layer->is_spiking && layer->reset_spike_counts) {
            layer->reset_spike_counts(layer);
            
            SpikingLayer *spiking = (SpikingLayer *)layer;
            if (spiking->total_spikes && spiking->num_neurons > 0) {
                memset(spiking->total_spikes, 0, spiking->num_neurons * sizeof(int));
            }
        }
    }
    
}
    

void apply_bidirectional_pruning(Network *network, BidirectionalPruningInfo *info) {
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        
        if (layer->layer_type == LAYER_CONV2D) {
            Conv2DLayer *conv = (Conv2DLayer *)layer;
            
            bool has_out_pruning = info->pruned_out_channels[i] > 0;
            bool has_in_pruning = info->pruned_in_channels[i] > 0;
            
            if (has_out_pruning || has_in_pruning) {
                apply_channel_compression(conv,info->inactive_out_channels[i], info->inactive_in_channels[i]);
            }
        }
    }
    
    printf("Bidirectional pruning with compression applied\n");
}

void print_bidirectional_stats(BidirectionalPruningInfo *info) {
    printf("\n=== BIDIRECTIONAL PRUNING SUMMARY ===\n");
    printf("Backward pruning: %.1f%% output channels removed\n", info->total_backward_reduction);
    printf("Forward pruning:  %.1f%% input channels removed\n", info->total_forward_reduction);
    printf("Combined effect:  ~%.1f%% computation reduction\n", info->combined_reduction);
    printf("Strategy: Direct Spiking↔Conv2D connections only\n");
    printf("============================================\n");
}

void free_bidirectional_pruning_info(BidirectionalPruningInfo *info) {
    if (info) {
        free(info->inactive_neurons_count);
        free(info->layer_activity_ratio);
        free(info->pruned_out_channels);
        free(info->pruned_in_channels);
        free(info->backward_connections);
        free(info->forward_connections);
        
        for (size_t i = 0; i < info->num_layers; i++) {
            if (info->inactive_out_channels[i]) free(info->inactive_out_channels[i]);
            if (info->inactive_in_channels[i]) free(info->inactive_in_channels[i]);
        }
        free(info->inactive_out_channels);
        free(info->inactive_in_channels);
        free(info);
    }
}

//                                         
// mmmmmmmm  mmmmmmmm    mmmm    mmmmmmmm 
// """##"""  ##""""""  m#""""#   """##""" 
//    ##     ##        ##m          ##    
//    ##     #######    "####m      ##    
//    ##     ##             "##     ##    
//    ##     ##mmmmmm  #mmmmm#"     ##    
 //   ""     """"""""   """""       ""  

void test_bidirectional_pruning(const char *architecture_path, const char *weights_path, 
                                     const char *dataset_path, int num_samples, 
                                     int spike_threshold, DatasetFormat format, 
                                     int input_width, int input_height, int no_channels) {
    
    printf("\n🔄 === TEST: BIDIRECTIONAL PRUNING ===\n");
    printf("Architecture: %s\n", architecture_path);
    printf("Weights: %s\n", weights_path);
    printf("Samples: %d analysis + %d test\n", num_samples, num_samples);
    printf("Threshold: %d spikes\n", spike_threshold);
    printf("=========================================\n\n");

    Network *network = initialize_network_from_file(architecture_path, input_width, input_height, no_channels);
    if (!network) return;
    load_weights_from_json(network, weights_path);
    
    int total_samples = num_samples * 2;
    Dataset *full_dataset = load_dataset(dataset_path, format, total_samples, false, true);
    if (!full_dataset) {
        free_network(network);
        return;
    }
    
    Dataset *test_dataset = (Dataset*)malloc(sizeof(Dataset));
    test_dataset->num_samples = full_dataset->num_samples / 2;
    test_dataset->input_channels = full_dataset->input_channels;
    test_dataset->input_width = full_dataset->input_width;
    test_dataset->input_height = full_dataset->input_height;
    test_dataset->num_classes = full_dataset->num_classes;
    test_dataset->samples = full_dataset->samples;

    Dataset *analysis_dataset = (Dataset*)malloc(sizeof(Dataset));
    analysis_dataset->num_samples = full_dataset->num_samples / 2;
    analysis_dataset->input_channels = full_dataset->input_channels;
    analysis_dataset->input_width = full_dataset->input_width;
    analysis_dataset->input_height = full_dataset->input_height;
    analysis_dataset->num_classes = full_dataset->num_classes;
    analysis_dataset->samples = &full_dataset->samples[num_samples / 2];

    printf("🎯 BASELINE: Test bez pruning...\n");
    clock_t start = clock();
    float baseline_acc = test(network, test_dataset);
    clock_t end = clock();
    double baseline_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("   Accuracy: %.2f%%, Time: %.3fs\n\n", baseline_acc, baseline_time);

    printf("🔍 ANALYSIS: Gathering spike activity...\n");
    reset_bidirectional_pruning(network);
    
    test(network, analysis_dataset);

    BidirectionalPruningInfo *info = analyze_bidirectional_activity(network, spike_threshold);
    
    printf("🔄 APPLYING BIDIRECTIONAL PRUNING...\n");
    apply_bidirectional_pruning(network, info);
    
    start = clock();
    float pruned_acc = test(network, test_dataset);
    end = clock();
    double pruned_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("\n🏆 === RESULTS ===\n");
    printf("Baseline Accuracy: %.2f%%\n", baseline_acc);
    printf("Pruned Accuracy:   %.2f%% (%.2f%% drop)\n", pruned_acc, baseline_acc - pruned_acc);
    printf("Baseline Time:     %.3fs\n", baseline_time);
    printf("Pruned Time:       %.3fs (%.2fx speedup)\n", pruned_time, baseline_time / pruned_time);
    printf("Forward reduction: %.1f%%\n", info->total_forward_reduction);
    printf("Backward reduction: %.1f%%\n", info->total_backward_reduction);
    printf("Combined reduction: %.1f%%\n", info->combined_reduction);
    
    free_bidirectional_pruning_info(info);
    free(test_dataset);
    free(analysis_dataset);
    free_dataset(full_dataset);
    free_network(network);
    
    printf("\n✅ Bidirectional pruning test completed!\n");
}