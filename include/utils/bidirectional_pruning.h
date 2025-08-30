#ifndef BIDIRECTIONAL_PRUNING_H
#define BIDIRECTIONAL_PRUNING_H

#include <stddef.h>
#include <stdbool.h>
#include "../network.h"
#include "../layers/spiking_layer.h"
#include "../layers/conv2d_layer.h"
#include "dataset_loader.h"

// Struktura dla bidirectional pruning (backward + forward)
typedef struct {
    size_t num_layers;
    
    // Statystyki aktywności neuronów
    int *inactive_neurons_count;
    float *layer_activity_ratio;
    
    // BACKWARD PRUNING: Spiking → Conv2d (output channels)
    bool **inactive_out_channels;    // Które out_channels usunąć z Conv2d
    size_t *pruned_out_channels;
    
    // FORWARD PRUNING: Conv2d → Spiking (input channels)  
    bool **inactive_in_channels;     // Które in_channels usunąć z Conv2d
    size_t *pruned_in_channels;
    
    // Mapowanie połączeń między warstwami
    int *backward_connections;       // conv_layer_idx → spiking_layer_idx  
    int *forward_connections;        // spiking_layer_idx → conv_layer_idx
    
    // Statystyki
    float total_backward_reduction;  // % redukcja przez backward pruning
    float total_forward_reduction;   // % redukcja przez forward pruning
    float combined_reduction;        // Łączna redukcja obliczeń
} BidirectionalPruningInfo;

// Główne funkcje
BidirectionalPruningInfo* analyze_bidirectional_activity(Network *network, int threshold);
void apply_bidirectional_pruning(Network *network, BidirectionalPruningInfo *info);
void reset_bidirectional_pruning(Network *network);

// Funkcje analizy
void analyze_backward_connections(Network *network, BidirectionalPruningInfo *info, int threshold);
void analyze_forward_connections(Network *network, BidirectionalPruningInfo *info);
void calculate_pruning_statistics(Network *network, BidirectionalPruningInfo *info);

// Funkcje testowe
void test_bidirectional_pruning(const char *architecture_path, const char *weights_path, 
                                     const char *dataset_path, int num_samples, 
                                     int spike_threshold, DatasetFormat format, 
                                     int input_width, int input_height, int no_channels);

// Funkcje pomocnicze
void print_bidirectional_stats(BidirectionalPruningInfo *info);
void free_bidirectional_pruning_info(BidirectionalPruningInfo *info);
void apply_channel_compression(Conv2DLayer *layer, bool *inactive_out, bool *inactive_in);

#endif // BIDIRECTIONAL_PRUNING_H