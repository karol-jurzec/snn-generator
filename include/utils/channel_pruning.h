#ifndef CHANNEL_PRUNING_H
#define CHANNEL_PRUNING_H

#include <stddef.h>
#include <stdbool.h>
#include "../network.h"
#include "../layers/spiking_layer.h"
#include "../layers/conv2d_layer.h"

typedef struct {
    size_t num_layers;
    int *inactive_neurons_count;
    bool **inactive_channels;        // Backward pruning masks
    size_t *pruned_channels_count;   // Backward pruning counts
    float sparsity_ratio;
    
    bool **inactive_in_channels;     // forward pruning masks
    size_t *pruned_in_channels_count; // forward pruning counts 
    
} PruningInfo;

// main methods
void reset_spike_counters(Network *network);
PruningInfo* analyze_network_activity(Network *network, int threshold);
void apply_channel_pruning(Network *network, PruningInfo *pruning_info);
void print_pruning_stats(PruningInfo *pruning_info);

// helpers
bool check_channel_inactive(SpikingLayer *spiking_layer, Conv2DLayer *conv_layer, 
                                         MaxPool2DLayer *pool_layer, int channel_idx, int threshold) ;
void mark_inactive_channels(Network *network, PruningInfo *pruning_info, int threshold);
void free_pruning_info(PruningInfo *pruning_info);

void test_channel_pruning(const char *architecture_path, const char *weights_path, const char *dataset_path,
     int num_samples_for_analysis, int spike_threshold, DatasetFormat format, int sampele_width, int sampele_height, int no_channels);

void reset_channel_pruning(Network *network);
void study_threshold_impact(const char *architecture_path, const char *weights_path, 
                           const char *dataset_path, const char *results_file, 
                           int num_samples_for_analysis);

#endif // CHANNEL_PRUNING_H