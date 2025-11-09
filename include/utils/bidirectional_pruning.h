#ifndef BIDIRECTIONAL_PRUNING_H
#define BIDIRECTIONAL_PRUNING_H

#include <stddef.h>
#include <stdbool.h>
#include "../network.h"
#include "../layers/spiking_layer.h"
#include "../layers/conv2d_layer.h"
#include "dataset_loader.h"

// bidirectional pruning struct (backward + forward)
typedef struct {
    size_t num_layers;
    
    // neuron activity stats
    int *inactive_neurons_count;
    float *layer_activity_ratio;
    
    // bckward pruning: Conv2d → [MaxPool2d] → Spiking
    bool **inactive_out_channels;        
    size_t *pruned_out_channels;         
    
    // forward pruning: Spiking → Conv2d
    bool **inactive_in_channels;         
    size_t *pruned_in_channels;          
    
    // CONNECTION MAPPING WITH MAXPOOL SUPPORT
    int *backward_connections;           // conv_idx → spiking_idx 
    int *forward_connections;            // spiking_idx → next_conv_idx
    int *maxpool_between_backward;       // maxpool_idx between conv→spiking (-1 if none)
    int *maxpool_between_forward;        // maxpool_idx between spiking→conv (-1 if none)
    
    // stats
    float total_backward_reduction;
    float total_forward_reduction;
    float combined_reduction;
    
} BidirectionalPruningInfo;

BidirectionalPruningInfo* analyze_bidirectional_activity(Network *network, int threshold);
void apply_bidirectional_pruning(Network *network, BidirectionalPruningInfo *info);
void reset_bidirectional_pruning(Network *network);

// analysis methods
void analyze_backward_connections(Network *network, BidirectionalPruningInfo *info, int threshold);
void analyze_forward_connections(Network *network, BidirectionalPruningInfo *info);
void calculate_pruning_statistics(Network *network, BidirectionalPruningInfo *info);

// test methods -- can be moved to test
void test_bidirectional_pruning(const char *architecture_path, const char *weights_path, 
                                     const char *dataset_path, int num_samples, 
                                     int spike_threshold, DatasetFormat format, 
                                     int input_width, int input_height, int no_channels);

// helpers 
void print_bidirectional_stats(BidirectionalPruningInfo *info);
void free_bidirectional_pruning_info(BidirectionalPruningInfo *info);
void apply_channel_compression(Conv2DLayer *layer, bool *inactive_out, bool *inactive_in);

#endif // BIDIRECTIONAL_PRUNING_H