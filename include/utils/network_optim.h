#include "../layers/spiking_layer.h"
#include "../layers/conv2d_layer.h"
#include "layer_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Struktura do przechowywania informacji o aktywności neuronów
typedef struct {
    int* inactive_neurons;  // Indeksy nieaktywnych neuronów
    size_t count;           // Liczba nieaktywnych neuronów
} NeuronActivity;

NeuronActivity analyze_spiking_activity(SpikingLayer* layer, int time_steps);

void optimize_conv2d_after_spiking(Conv2DLayer* conv_layer, NeuronActivity activity);
void optimize_conv2d_before_spiking(Conv2DLayer* conv_layer, NeuronActivity activity);


// helpers 

int* find_unused_filters(Conv2DLayer* layer, NeuronActivity activity);
void prune_conv2d_layer(Conv2DLayer* layer, int* unused_filters); 

int count_unused(const int* unused_mask, int size);
int* find_unused_input_channels(Conv2DLayer* layer, NeuronActivity activity);
void prune_conv2d_input_channels(Conv2DLayer* layer, int* unused_channels);

