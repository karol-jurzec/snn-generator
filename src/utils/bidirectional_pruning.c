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
    
    // Connection mapping
    info->backward_connections = (int *)malloc(network->num_layers * sizeof(int));
    info->forward_connections = (int *)malloc(network->num_layers * sizeof(int));
    
    // Initialize connection arrays to -1 (no connection)
    for (size_t i = 0; i < network->num_layers; i++) {
        info->backward_connections[i] = -1;
        info->forward_connections[i] = -1;
        info->inactive_out_channels[i] = NULL;
        info->inactive_in_channels[i] = NULL;
    }
    
    // Alokuj maski tylko dla warstw conv2d
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

void map_layer_connections(Network *network, BidirectionalPruningInfo *info) {
    printf("\n=== MAPOWANIE POŁĄCZEŃ MIĘDZY WARSTWAMI ===\n");
    
    for (size_t i = 0; i < network->num_layers - 1; i++) {
        LayerBase *current_layer = network->layers[i];
        LayerBase *next_layer = network->layers[i + 1];
        
        // BACKWARD: Conv2d → Spiking (dla backward pruning)
        if (current_layer->layer_type == LAYER_CONV2D && next_layer->is_spiking) {
            info->backward_connections[i] = i + 1;  // conv[i] → spiking[i+1]
            printf("📍 Backward connection: Conv2D[%zu] → Spiking[%zu]\n", i, i+1);
        }
        
        // FORWARD: Spiking → Conv2d (dla forward pruning)
        if (current_layer->is_spiking && next_layer->layer_type == LAYER_CONV2D) {
            info->forward_connections[i] = i + 1;   // spiking[i] → conv[i+1]
            printf("📍 Forward connection: Spiking[%zu] → Conv2D[%zu]\n", i, i+1);
        }
    }
    printf("✅ Mapowanie połączeń zakończone\n");
}

void analyze_backward_connections(Network *network, BidirectionalPruningInfo *info, int threshold) {
    printf("\n--- BACKWARD PRUNING: Conv2D → Spiking Analysis ---\n");
    
    for (size_t i = 0; i < network->num_layers; i++) {
        if (info->backward_connections[i] != -1) {
            size_t spiking_idx = info->backward_connections[i];
            
            LayerBase *conv_layer = network->layers[i];
            LayerBase *spiking_layer = network->layers[spiking_idx];
            
            Conv2DLayer *conv = (Conv2DLayer *)conv_layer;
            SpikingLayer *spiking = (SpikingLayer *)spiking_layer;
            
            printf("🔍 Analyzing Conv2D[%zu] → Spiking[%zu]\n", i, spiking_idx);
            printf("   Conv2D: %d→%d channels, %dx%d input\n", 
                   conv->in_channels, conv->out_channels, conv->input_dim, conv->input_dim);
            printf("   Spiking: %zu neurons\n", spiking->num_neurons);
            
            // Oblicz wymiary po conv2d (bez MaxPool!)
            int conv_out_h = (conv->input_dim + 2*conv->padding - conv->kernel_size) / conv->stride + 1;
            int conv_out_w = conv_out_h;
            int neurons_per_channel = conv_out_h * conv_out_w;
            
            printf("   Mapping: %dx%d=%d neurons per channel\n", 
                   conv_out_h, conv_out_w, neurons_per_channel);
            
            // Sprawdź każdy kanał
            for (int channel = 0; channel < conv->out_channels; channel++) {
                bool channel_inactive = true;
                int active_neurons_in_channel = 0;
                
                // Sprawdź wszystkie neurony dla tego kanału
                for (int h = 0; h < conv_out_h; h++) {
                    for (int w = 0; w < conv_out_w; w++) {
                        int neuron_idx = channel * neurons_per_channel + h * conv_out_w + w;
                        
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
                    printf("   ❌ Channel %d: inactive (%d spikes total)\n", channel, 0);
                } else {
                    printf("   ✅ Channel %d: active (%d neurons spiking)\n", channel, active_neurons_in_channel);
                }
            }
            
            float pruning_ratio = (float)info->pruned_out_channels[i] / conv->out_channels * 100.0f;
            printf("   📊 Result: %zu/%d channels pruned (%.1f%%)\n", 
                   info->pruned_out_channels[i], conv->out_channels, pruning_ratio);
        }
    }
}

void analyze_forward_connections(Network *network, BidirectionalPruningInfo *info) {
    printf("\n--- FORWARD PRUNING: Spiking → Conv2D Propagation ---\n");
    
    for (size_t i = 0; i < network->num_layers; i++) {
        if (info->forward_connections[i] != -1) {
            size_t conv_idx = info->forward_connections[i];
            
            LayerBase *spiking_layer = network->layers[i];
            LayerBase *conv_layer = network->layers[conv_idx];
            
            SpikingLayer *spiking = (SpikingLayer *)spiking_layer;
            Conv2DLayer *conv = (Conv2DLayer *)conv_layer;
            
            printf("🔄 Propagating: Spiking[%zu] → Conv2D[%zu]\n", i, conv_idx);
            
            // Znajdź poprzednią conv2d warstwę która produkuje input dla tej spiking
            Conv2DLayer *prev_conv = NULL;
            for (int j = i - 1; j >= 0; j--) {
                if (network->layers[j]->layer_type == LAYER_CONV2D) {
                    prev_conv = (Conv2DLayer *)network->layers[j];
                    printf("   Input pochodzi z Conv2D[%d]: %d channels\n", j, prev_conv->out_channels);
                    
                    // Sprawdź czy poprzednia conv2d ma inactive channels
                    if (info->inactive_out_channels[j]) {
                        // Propaguj te same maski jako input channels
                        size_t propagated_count = 0;
                        
                        for (int ch = 0; ch < prev_conv->out_channels && ch < conv->in_channels; ch++) {
                            if (info->inactive_out_channels[j][ch]) {
                                info->inactive_in_channels[conv_idx][ch] = true;
                                propagated_count++;
                            }
                        }
                        
                        info->pruned_in_channels[conv_idx] = propagated_count;
                        
                        printf("   ✅ Propagated %zu inactive channels to Conv2D[%zu] input\n", 
                               propagated_count, conv_idx);
                    } else {
                        printf("   ℹ️  No inactive channels to propagate from Conv2D[%d]\n", j);
                    }
                    break;
                }
            }
            
            if (!prev_conv) {
                printf("   ⚠️  Warning: No previous Conv2D found for Spiking[%zu]\n", i);
            }
        }
    }
}

BidirectionalPruningInfo* analyze_bidirectional_activity(Network *network, int threshold) {
    printf("\n🔄 === BIDIRECTIONAL PRUNING ANALYSIS ===\n");
    printf("Threshold: %d spikes\n", threshold);
    
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        if (layer->is_spiking) {
            SpikingLayer *spiking = (SpikingLayer *)layer;
            if (spiking->total_spikes) {
                for (size_t j = 0; j < spiking->num_neurons; j++) {
                    if (spiking->total_spikes[j] > 0) {
                        break;
                    }
                }
            }
        }
    }
    
    BidirectionalPruningInfo *info = create_bidirectional_info(network);
    
    map_layer_connections(network, info);
    analyze_backward_connections(network, info, threshold);
    analyze_forward_connections(network, info);
    calculate_pruning_statistics(network, info);
    
    return info;
}

void calculate_pruning_statistics(Network *network, BidirectionalPruningInfo *info) {
    printf("\n--- COMPUTING PRUNING STATISTICS ---\n");
    
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
            
            printf("Conv2D[%zu]: out %zu/%d pruned, in %zu/%d pruned\n",
                   i, info->pruned_out_channels[i], conv->out_channels,
                   info->pruned_in_channels[i], conv->in_channels);
        }
    }
    
    info->total_backward_reduction = (float)total_pruned_out / total_out_channels * 100.0f;
    info->total_forward_reduction = (float)total_pruned_in / total_in_channels * 100.0f;
    
    // Combined reduction (aproximadamente multiplicative)
    float backward_efficiency = 1.0f - info->total_backward_reduction / 100.0f;
    float forward_efficiency = 1.0f - info->total_forward_reduction / 100.0f;
    info->combined_reduction = (1.0f - backward_efficiency * forward_efficiency) * 100.0f;
    
    printf("\n📊 PRUNING STATISTICS:\n");
    printf("   Backward reduction: %.1f%% (%zu/%zu out_channels)\n", 
           info->total_backward_reduction, total_pruned_out, total_out_channels);
    printf("   Forward reduction:  %.1f%% (%zu/%zu in_channels)\n", 
           info->total_forward_reduction, total_pruned_in, total_in_channels);
    printf("   Combined reduction: ~%.1f%% (theoretical)\n", info->combined_reduction);
}

void reset_bidirectional_pruning(Network *network) {
    printf("Resetowanie bidirectional pruning...\n");
    
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        
        if (layer->layer_type == LAYER_CONV2D) {
            Conv2DLayer *conv = (Conv2DLayer *)layer;
            
            // 🔄 RESET CHANNEL COMPRESSION - przywróć oryginalne wymiary
            if (conv->out_active_channels_idx) {
                free(conv->out_active_channels_idx);
                conv->out_active_channels_idx = NULL;
            }
            
            if (conv->in_active_channels_idx) {
                free(conv->in_active_channels_idx);
                conv->in_active_channels_idx = NULL;
            }
            
            // Przywróć oryginalne wymiary
            conv->out_channels = conv->original_out_channels;
            conv->in_channels = conv->original_in_channels;
            
        }
    }
    
    // Reset spike counters (bez zmian)
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
    
    printf("✅ Bidirectional pruning reset completed\n");
}
    

void apply_bidirectional_pruning(Network *network, BidirectionalPruningInfo *info) {
    printf("\n=== APLIKOWANIE BIDIRECTIONAL PRUNING Z COMPRESSION ===\n");
    
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        
        if (layer->layer_type == LAYER_CONV2D) {
            Conv2DLayer *conv = (Conv2DLayer *)layer;
            
            // Sprawdź czy są zmiany do zastosowania
            bool has_out_pruning = info->pruned_out_channels[i] > 0;
            bool has_in_pruning = info->pruned_in_channels[i] > 0;
            
            if (has_out_pruning || has_in_pruning) {
                printf("  Conv2D[%zu]: Applying compression...\n", i);
                
                // 🚀 UŻYJ NOWEJ FUNKCJI COMPRESSION zamiast memcpy masek
                apply_channel_compression(conv, 
                                        info->inactive_out_channels[i], 
                                        info->inactive_in_channels[i]);
                
                printf("    ✅ Compression applied: %zu out + %zu in channels pruned\n", 
                       info->pruned_out_channels[i], info->pruned_in_channels[i]);
            }
        }
    }
    
    printf("✅ Bidirectional pruning with compression applied\n");
}

void print_bidirectional_stats(BidirectionalPruningInfo *info) {
    printf("\n📊 === BIDIRECTIONAL PRUNING SUMMARY ===\n");
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

    // Wczytaj sieć
    Network *network = initialize_network_from_file(architecture_path, input_width, input_height, no_channels);
    if (!network) return;
    load_weights_from_json(network, weights_path);
    
    // Wczytaj i podziel dataset (pierwszy num_samples do testu, drugi do analizy)
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

    // BASELINE: Test bez pruning
    printf("🎯 BASELINE: Test bez pruning...\n");
    clock_t start = clock();
    float baseline_acc = test(network, test_dataset);
    clock_t end = clock();
    double baseline_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("   Accuracy: %.2f%%, Time: %.3fs\n\n", baseline_acc, baseline_time);

    // Przeprowadź analizę aktywności
    printf("🔍 ANALYSIS: Gathering spike activity...\n");
    reset_bidirectional_pruning(network);
    
    test(network, analysis_dataset);

    BidirectionalPruningInfo *info = analyze_bidirectional_activity(network, spike_threshold);
    
    // Apply bidirectional pruning
    printf("🔄 APPLYING BIDIRECTIONAL PRUNING...\n");
    apply_bidirectional_pruning(network, info);
    
    // Test z bidirectional pruning
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
    
    // Cleanup
    free_bidirectional_pruning_info(info);
    free(test_dataset);
    free(analysis_dataset);
    free_dataset(full_dataset);
    free_network(network);
    
    printf("\n✅ Bidirectional pruning test completed!\n");
}