#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../../include/utils/pruning_utils.h"
#include "../../include/utils/channel_pruning.h"
#include "../../include/layers/layer_base.h"
#include "../../include/utils/network_loader.h"
#include "../../include/utils/perf.h"
#include "../../include/models/lif_neuron.h"

void reset_spike_counters(Network *network) {
    printf("Spike counter reset...\n");
    
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        
        // reset compression dla Conv2D
        if (layer->layer_type == LAYER_CONV2D) {
            Conv2DLayer *conv = (Conv2DLayer *)layer;
            
            // reset compression
            if (conv->out_active_channels_idx) {
                free(conv->out_active_channels_idx);
                conv->out_active_channels_idx = NULL;
            }
            
            if (conv->in_active_channels_idx) {
                free(conv->in_active_channels_idx);
                conv->in_active_channels_idx = NULL;
            }
            
            // restore original 
            conv->out_channels = conv->original_out_channels;
            conv->in_channels = conv->original_in_channels;
        }
        
        // reset spike counters
        if (layer->is_spiking && layer->reset_spike_counts) {
            layer->reset_spike_counts(layer);
            
            SpikingLayer *spiking = (SpikingLayer *)layer;
            if (spiking->total_spikes && spiking->num_neurons > 0) {
                memset(spiking->total_spikes, 0, spiking->num_neurons * sizeof(int));
            }
        }
    }
    
    printf("Reset completed\n");
}

PruningInfo* create_pruning_info(Network *network) {
    PruningInfo *info = (PruningInfo *)malloc(sizeof(PruningInfo));
    info->num_layers = network->num_layers;
    info->inactive_neurons_count = (int *)calloc(network->num_layers, sizeof(int));
    info->inactive_channels = (bool **)malloc(network->num_layers * sizeof(bool *));
    info->pruned_channels_count = (size_t *)calloc(network->num_layers, sizeof(size_t));
    info->sparsity_ratio = 0.0f;
    
    // init conv2d masks
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        if (layer->layer_type == LAYER_CONV2D) {
            Conv2DLayer *conv = (Conv2DLayer *)layer;
            info->inactive_channels[i] = (bool *)calloc(conv->out_channels, sizeof(bool));
        } else {
            info->inactive_channels[i] = NULL;
        }
    }
    
    return info;
}

PruningInfo* analyze_network_activity(Network *network, int threshold) {
    printf("=== Network activity analysis (threshold: %d) ===\n", threshold);
    
    PruningInfo *info = create_pruning_info(network);
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        
        if (layer->is_spiking) {
            SpikingLayer *spiking = (SpikingLayer *)layer;
            
            // inactive neuron counter 
            int inactive_count = 0;
            for (size_t j = 0; j < spiking->num_neurons; j++) {
                if (spiking->total_spikes[j] <= threshold) {
                    inactive_count++;
                }
            }
            info->inactive_neurons_count[i] = inactive_count;
            
            printf("Layer %zu (Spiking): %d/%zu inactive neurons\n", 
                   i, inactive_count, spiking->num_neurons);
        }
    }
    
    mark_inactive_channels(network, info, threshold);
    
    return info;
}

void mark_inactive_channels(Network *network, PruningInfo *info, int threshold) {
    printf("\n=== Conv channels analysis ===\n");

    // iterate num_layers - 2 to preven OOR exception 
    // and because it is imposible to have two spiking layers one after another 

    for (size_t i = 0; i < network->num_layers - 2; i++) {  
        LayerBase *conv_layer = network->layers[i];
        LayerBase *next_layer = network->layers[i + 1];
        LayerBase *spiking_layer = network->layers[i + 2];
        
        // check pattern: Conv2D → MaxPool2D → SpikingLayer
        if (conv_layer->layer_type == LAYER_CONV2D && 
            next_layer->layer_type == LAYER_MAXPOOL2D && 
            spiking_layer->is_spiking) {
            
            Conv2DLayer *conv = (Conv2DLayer *)conv_layer;
            MaxPool2DLayer *pool = (MaxPool2DLayer *)next_layer;
            SpikingLayer *spiking = (SpikingLayer *)spiking_layer;
            
            printf("Pattern found: Conv2D[%zu] → MaxPool2D[%zu] → Spiking[%zu]\n", i, i+1, i+2);
            
            for (int channel = 0; channel < conv->out_channels; channel++) {
                bool channel_inactive = check_channel_inactive(spiking, conv, pool, channel, threshold);
                
                if (channel_inactive) {
                    info->inactive_channels[i][channel] = true;
                    info->pruned_channels_count[i]++;
                    printf("  Channel %d: inactive \n", channel);
                } else {
                    printf("  Channel %d: active\n", channel);
                }
            }
            
            float pruning_ratio = (float)info->pruned_channels_count[i] / conv->out_channels * 100.0f;
            printf("Layer %zu: %zu/%d chanels will be pruned (%.1f%%)\n", 
                   i, info->pruned_channels_count[i], conv->out_channels, pruning_ratio);
        }
        
        // if Conv2D -> SpikingLayer
        if (conv_layer->layer_type == LAYER_CONV2D && next_layer->is_spiking) {
            Conv2DLayer *conv = (Conv2DLayer *)conv_layer;
            SpikingLayer *spiking = (SpikingLayer *)next_layer;
            
            for (int channel = 0; channel < conv->out_channels; channel++) {
                bool channel_inactive = check_channel_inactive(spiking, conv, NULL, channel, threshold);
                
                if (channel_inactive) {
                    info->inactive_channels[i][channel] = true;
                    info->pruned_channels_count[i]++;
                    printf("  Channel %d: inactive\n", channel);
                } else {
                    printf("  Channel %d: active\n", channel);
                }
            }
            
            printf("Layer %zu: %zu/%d channels will be prunned\n", i, info->pruned_channels_count[i], conv->out_channels);
            
            for (int channel = 0; channel < conv->out_channels; channel++) {
                bool channel_inactive = check_channel_inactive(spiking, conv, NULL, channel, threshold);
                
                if (channel_inactive) {
                    info->inactive_channels[i][channel] = true;
                    info->pruned_channels_count[i]++;
                    printf("Channel %d: inactive\n", channel);
                } else {
                    printf("Channel %d: active\n", channel);
                }
            }
            
            printf("RESULT: %zu/%d channels will be pruned (%.1f%%)\n", 
                   info->pruned_channels_count[i], conv->out_channels,
                   (float)info->pruned_channels_count[i] / conv->out_channels * 100.0f);
        }
    }
}

bool check_channel_inactive(SpikingLayer *spiking_layer, Conv2DLayer *conv_layer, 
                                         MaxPool2DLayer *pool_layer, int channel_idx, int threshold)  {
    // out dimms conv2d
    int conv_output_h = (conv_layer->input_dim + 2*conv_layer->padding - conv_layer->kernel_size) 
                   / conv_layer->stride + 1;
    int conv_output_w = conv_output_h; // square input assumption

    // if between conv2d and spiking layer is pooling output dimms 
    // are overwriten dimms after pooling

    int output_h, output_w; 

    if (pool_layer) {
        output_h = conv_output_h / pool_layer->kernel_size;
        output_w = conv_output_w / pool_layer->kernel_size;
    } else {
        output_h = conv_output_h;
        output_w = conv_output_w;
    }

    int neurons_per_channel = output_h * output_w;
    
    for (int h = 0; h < output_h; h++) {
        for (int w = 0; w < output_w; w++) {
            int neuron_idx = channel_idx * neurons_per_channel + h * output_w + w;
            
            if (neuron_idx < spiking_layer->num_neurons) {
                if (spiking_layer->total_spikes[neuron_idx] > threshold) {
                    return false;  // active channel found
                }
            }
        }
    }
    
    return true;  
}


void apply_channel_pruning(Network *network, PruningInfo *pruning_info) {
    printf("\n=== Channel pruning with compression ===\n");
    
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        
        if (layer->layer_type == LAYER_CONV2D && pruning_info->inactive_channels[i]) {
            Conv2DLayer *conv = (Conv2DLayer *)layer;
            
            if (pruning_info->pruned_channels_count[i] > 0) {
                printf("Layer %zu: Applying compression...\n", i);
                
                apply_channel_compression(conv, 
                                        pruning_info->inactive_channels[i],  // backward mask
                                        NULL);                              // forward mask = NULL
                
                printf("Compression applied: %zu out_channels pruned\n", 
                       pruning_info->pruned_channels_count[i]);
            }
        }
    }
    
    printf("Channel pruning with compression applied\n");
}

void print_pruning_stats(PruningInfo *pruning_info) {
    printf("\n=== Pruning stats ===\n");
    
    size_t total_pruned = 0;
    size_t total_channels = 0;
    
    for (size_t i = 0; i < pruning_info->num_layers; i++) {
        if (pruning_info->pruned_channels_count[i] > 0) {
            total_pruned += pruning_info->pruned_channels_count[i];
        }
    }
    
    printf("Totaly %zu pruned channels\n", total_pruned);
}

void free_pruning_info(PruningInfo *pruning_info) {
    if (pruning_info) {
        free(pruning_info->inactive_neurons_count);
        free(pruning_info->pruned_channels_count);
        
        for (size_t i = 0; i < pruning_info->num_layers; i++) {
            if (pruning_info->inactive_channels[i]) {
                free(pruning_info->inactive_channels[i]);
            }
        }
        free(pruning_info->inactive_channels);
        free(pruning_info);
    }
}

//                                          
// mmmmmmmm  mmmmmmmm    mmmm    mmmmmmmm 
// """##"""  ##""""""  m#""""#   """##""" 
//    ##     ##        ##m          ##    
//    ##     #######    "####m      ##    
//    ##     ##             "##     ##    
//    ##     ##mmmmmm  #mmmmm#"     ##    
//    ""     """"""""   """""       ""  

void test_channel_pruning(const char *architecture_path, const char *weights_path, 
                         const char *dataset_path, int num_samples_for_analysis,
                         int spike_threshold, DatasetFormat format, int input_width, int input_height, 
                         int no_channels) {
    printf("\n==========================================\n");
    printf("STARTING CHANNEL PRUNING TEST\n");
    printf("==========================================\n");
    printf("Architecture: %s\n", architecture_path);
    printf("Weights: %s\n", weights_path);  
    printf("Dataset: %s\n", dataset_path);
    printf("Samples for analysis: %d\n", num_samples_for_analysis);
    printf("Spike threshold: %d\n", spike_threshold);
    printf("==========================================\n\n");

    // STEP 1: Load network
    printf("STEP 1: Loading network...\n");
    Network *network = initialize_network_from_file(architecture_path, input_width, input_height, no_channels);
    if (!network) {
        printf("Error: Failed to load network!\n");
        return;
    }
    load_weights_from_json(network, weights_path);
    printf("Network loaded successfully\n\n");

    // STEP 2: Load dataset and split into two parts
    printf("STEP 2: Loading and splitting dataset...\n");
    int total_samples = num_samples_for_analysis * 2;
    Dataset *full_dataset = load_dataset(dataset_path, format, total_samples, false, false);
    if (!full_dataset) {
        printf("Error: Failed to load dataset!\n");
        if (network) free_network(network);
        return;
    }

    // Create test dataset (first half)
    Dataset *test_dataset = (Dataset*)malloc(sizeof(Dataset));
    test_dataset->num_samples = num_samples_for_analysis;
    test_dataset->input_channels = full_dataset->input_channels;
    test_dataset->input_width = full_dataset->input_width;
    test_dataset->input_height = full_dataset->input_height;
    test_dataset->num_classes = full_dataset->num_classes;
    test_dataset->samples = full_dataset->samples;

    // Create analysis dataset (second half)
    Dataset *analysis_dataset = (Dataset*)malloc(sizeof(Dataset));
    analysis_dataset->num_samples = num_samples_for_analysis;
    analysis_dataset->input_channels = full_dataset->input_channels;
    analysis_dataset->input_width = full_dataset->input_width;
    analysis_dataset->input_height = full_dataset->input_height;
    analysis_dataset->num_classes = full_dataset->num_classes;
    analysis_dataset->samples = &full_dataset->samples[num_samples_for_analysis];

    printf("Dataset split completed:\n");
    printf("   - Test dataset: samples 0-%d (%zu samples)\n", num_samples_for_analysis-1, test_dataset->num_samples);
    printf("   - Analysis dataset: samples %d-%d (%zu samples)\n", 
           num_samples_for_analysis, total_samples-1, analysis_dataset->num_samples);
    printf("\n");

    // STEP 3: Test accuracy BEFORE pruning
    printf("STEP 3: Testing accuracy BEFORE pruning (on test dataset)...\n");
    clock_t start_time = clock();
    float accuracy_before = test(network, test_dataset);
    clock_t end_time = clock();
    double inference_time_before = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Accuracy before pruning: %.2f%% (time: %.3fs)\n\n", accuracy_before, inference_time_before);

    // STEP 4: Reset counters and analyze neuron activity
    printf("STEP 4: Analyzing neuron activity (on analysis dataset)...\n");
    reset_spike_counters(network);
    printf("Passing %zu samples through network for activity analysis...\n", analysis_dataset->num_samples);
    test(network, analysis_dataset);
    printf("Activity analysis completed\n\n");

    // STEP 5: Identify channels for pruning
    printf("STEP 5: Identifying inactive channels (based on analysis dataset)...\n");
    PruningInfo *pruning_info = analyze_network_activity(network, spike_threshold);
    print_pruning_stats(pruning_info);

    // STEP 6: Apply pruning
    printf("\nSTEP 6: Applying pruning...\n");
    apply_channel_pruning(network, pruning_info);
    printf("Pruning applied\n\n");

    // STEP 7: Test accuracy AFTER pruning
    printf("STEP 7: Testing accuracy AFTER pruning (on test dataset)...\n");
    start_time = clock();
    float accuracy_after = test(network, test_dataset);
    end_time = clock();
    double inference_time_after = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Accuracy after pruning: %.2f%% (time: %.3fs)\n\n", accuracy_after, inference_time_after);

    // STEP 8: Summary of results
    printf("==========================================\n");
    printf("CHANNEL PRUNING RESULTS SUMMARY\n");
    printf("==========================================\n");
    printf("METHOD:\n");
    printf("- Analysis dataset: samples %d-%d (%zu samples)\n", 
           num_samples_for_analysis, total_samples-1, analysis_dataset->num_samples);
    printf("- Test dataset: samples 0-%d (%zu samples)\n", 
           num_samples_for_analysis-1, test_dataset->num_samples);
    printf("- Spike threshold: %d\n", spike_threshold);
    printf("\nRESULTS:\n");
    printf("Accuracy BEFORE pruning: %.2f%%\n", accuracy_before);
    printf("Accuracy AFTER pruning: %.2f%%\n", accuracy_after);
    printf("Accuracy change: %+.2f%% ", accuracy_after - accuracy_before);
    if (accuracy_after - accuracy_before > -1.0f) {
        printf("(acceptable)\n");
    } else {
        printf("(significant drop)\n");
    }
    printf("Inference time BEFORE: %.3fs\n", inference_time_before);
    printf("Inference time AFTER: %.3fs\n", inference_time_after);
    printf("Speedup: %.2fx ", inference_time_before / inference_time_after);
    if (inference_time_after < inference_time_before) {
        printf("(speedup)\n");
    } else {
        printf("(slowdown)\n");
    }
    printf("==========================================\n");

    // Cleanup
    free_pruning_info(pruning_info);
    free(test_dataset);
    free(analysis_dataset);
    free_dataset(full_dataset);
    free_network(network);

    printf("Channel pruning test completed successfully\n");
}

void reset_channel_pruning(Network *network) {
    printf("Resetting channel pruning and compression...\n");
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
            printf("  Conv2D[%zu]: Reset - back to %d out_channels, %d in_channels\n", 
                   i, conv->out_channels, conv->in_channels);
        }
    }
    printf("Channel pruning reset completed\n");
}
