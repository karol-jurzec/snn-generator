
#include "../../include/utils/network_optim.h"

int compare_ints(const void* a, const void* b) {
    int arg1 = *(const int*)a;
    int arg2 = *(const int*)b;
    return (arg1 > arg2) - (arg1 < arg2);
}

NeuronActivity analyze_spiking_activity(SpikingLayer* layer, int time_steps) {
    NeuronActivity activity;
    activity.inactive_neurons = malloc(layer->base.output_size * sizeof(int));
    activity.count = 0;


    
    for (size_t i = 0; i < layer->base.output_size; i++) {
        if (layer->total_spikes[i] == 0) {
            activity.inactive_neurons[activity.count++] = i;
        }
    }
    
    return activity;
}

void optimize_conv2d_after_spiking(Conv2DLayer* conv_layer, NeuronActivity activity) {
    if (conv_layer->base.layer_type != LAYER_CONV2D) return;

    // 1. Znajdź nieużywane filtry (teraz z uwzględnieniem mapowania przestrzennego)
    int* unused_filters = find_unused_filters(conv_layer, activity);
    int unused_count = count_unused(unused_filters, conv_layer->out_channels);
    
    if (unused_count > 0) {
        printf("Pruning %d/%d filters in Conv2D layer\n", 
               unused_count, conv_layer->out_channels);
        
        // 2. Przeprowadź rzeczywisty pruning
        prune_conv2d_layer(conv_layer, unused_filters);
        
        // 3. Aktualizacja gradientów inputów
        // if (conv_layer->base.input_gradients) {
        //     free(conv_layer->base.input_gradients);
        //     conv_layer->base.input_gradients = malloc(conv_layer->base.num_inputs * sizeof(float));
        // }
    }
    
    free(unused_filters);
}

// Funkcja optymalizująca warstwę Conv2D przed warstwą spikingową
void optimize_conv2d_before_spiking(Conv2DLayer* conv_layer, NeuronActivity activity) {
    int size = 0;
    // 1. Znajdź nieużywane kanały wejściowe
    int* unused_channels = find_unused_input_channels(conv_layer, activity);
    
    // 2. Zaktualizuj rozmiary warstwy
    int new_in_channels = conv_layer->in_channels - count_unused(unused_channels, conv_layer->in_channels);
    
    // 3. Przeprowadź rzeczywisty pruning
    prune_conv2d_input_channels(conv_layer, unused_channels);
    
    free(unused_channels);
}

int* find_unused_filters(Conv2DLayer* layer, NeuronActivity activity) {
    int* unused_filters = calloc(layer->out_channels, sizeof(int));
    int spatial_size = calculate_output_dim(layer->input_dim, layer->kernel_size, 
                                         layer->stride, layer->padding);
    spatial_size *= spatial_size; // width * height
    
    // Sortuj nieaktywne neurony dla lepszego wyszukiwania
    qsort(activity.inactive_neurons, activity.count, sizeof(int), compare_ints);
    
    #pragma omp parallel for
    for (int f = 0; f < layer->out_channels; f++) {
        int filter_start = f * spatial_size;
        int filter_end = (f + 1) * spatial_size;
        
        // Sprawdź czy wszystkie neurony filtra są nieaktywne
        int all_inactive = 1;
        for (int n = filter_start; n < filter_end; n++) {
            if (!bsearch(&n, activity.inactive_neurons, activity.count, 
                         sizeof(int), compare_ints)) {
                all_inactive = 0;
                break;
            }
        }
        
        unused_filters[f] = all_inactive;
    }
    
    return unused_filters;
}

// Przeprowadź pruning warstwy Conv2D
void prune_conv2d_layer(Conv2DLayer* layer, int* unused_filters) {
    int new_out_channels = layer->out_channels - count_unused(unused_filters, layer->out_channels);
    if (new_out_channels == 0) {
        printf("Warning: All filters would be pruned! Skipping.\n");
        return;
    }

    for(int c = 0; c < layer->out_channels; ++c) {
        if(unused_filters[c]) {
            layer->deactive_out_channels[c] = true;
        }
    }

    // int kernel_elems = layer->kernel_size * layer->kernel_size;
    // size_t new_weight_size = layer->in_channels * new_out_channels * kernel_elems;
    
    // // 1. Alokacja nowych wag i biasów
    // float* new_weights = malloc(new_weight_size * sizeof(float));
    // float* new_biases = malloc(new_out_channels * sizeof(float));
    // float* new_weight_grads = malloc(new_weight_size * sizeof(float));
    
    // // 2. Przepisywanie tylko aktywnych filtrów
    // int new_f_idx = 0;
    // for (int old_f_idx = 0; old_f_idx < layer->out_channels; old_f_idx++) {
    //     if (!unused_filters[old_f_idx]) {
    //         // Kopiuj wagi dla całego filtra
    //         memcpy(&new_weights[new_f_idx * layer->in_channels * kernel_elems],
    //                &layer->base.weights[old_f_idx * layer->in_channels * kernel_elems],
    //                layer->in_channels * kernel_elems * sizeof(float));
            
    //         // Kopiuj bias i gradienty
    //         new_biases[new_f_idx] = layer->biases[old_f_idx];
            
    //         // if (layer->base.weight_gradients) {
    //         //     memcpy(&new_weight_grads[new_f_idx * layer->in_channels * kernel_elems],
    //         //            &layer->base.weight_gradients[old_f_idx * layer->in_channels * kernel_elems],
    //         //            layer->in_channels * kernel_elems * sizeof(float));
    //         // }
            
    //         new_f_idx++;
    //     }
    // }
    
    // // 3. Aktualizacja struktury warstwy
    // free(layer->base.weights);
    // free(layer->biases);
    // // if (layer->base.weight_gradients) free(layer->base.weight_gradients);
    
    // layer->base.weights = new_weights;
    // layer->biases = new_biases;
    // // layer->base.weight_gradients = new_weight_grads;
    // layer->out_channels = new_out_channels;
    
    // // 4. Aktualizacja wymiarów wyjściowych
    // int output_dim = calculate_output_dim(layer->input_dim, layer->kernel_size, 
    //                                    layer->stride, layer->padding);
    // layer->base.output_size = output_dim * output_dim * new_out_channels;
    
    // // 5. Realokacja output buffer jeśli potrzebne
    // if (layer->base.output_size != output_dim * output_dim * layer->out_channels) {
    //     free(layer->base.output);
    //     layer->base.output = malloc(layer->base.output_size * sizeof(float));
    // }
}

int count_unused(const int* unused_mask, int size) {
    int count = 0;
    for (int i = 0; i < size; i++) {
        if (unused_mask[i]) {
            count++;
        }
    }
    return count;
}

/**
 * Znajduje nieużywane kanały wejściowe w warstwie Conv2D na podstawie aktywności neuronów w poprzedniej warstwie
 * @param layer Warstwa Conv2D do analizy
 * @param activity Informacja o aktywności neuronów w poprzedniej warstwie
 * @return Maska nieużywanych kanałów (1 - nieużywany, 0 - używany)
 */

int* find_unused_input_channels(Conv2DLayer* layer, NeuronActivity activity) {
    int* unused_channels = (int*)calloc(layer->in_channels, sizeof(int));
    if (!unused_channels) {
        fprintf(stderr, "Memory allocation failed in find_unused_input_channels\n");
        exit(EXIT_FAILURE);
    }

     // 2 output channels 2 x 2
    // total_spike_counts=[0, 0, 0, 0, 35, 35, 0, 2]
    // inactive_neurons=[0, 1, 2, 3, 6]

    // Zakładamy, że poprzednia warstwa ma neurons_per_channel = output_width * output_height
    int neurons_per_channel = ( layer->input_dim * layer->input_dim );
    
    // Sprawdź każdy kanał wejściowy
    for (int ch = 0; ch < layer->in_channels; ch++) {
        int is_channel_active = 0;
        int start_neuron = ch * neurons_per_channel;
        int end_neuron = (ch + 1) * neurons_per_channel;

        // Sprawdź czy jakikolwiek neuron w tym kanale jest aktywny
        for (int n = start_neuron; n < end_neuron && !is_channel_active; n++) {
            int is_neuron_inactive = 0;
            
            // Sprawdź czy neuron jest na liście nieaktywnych
            for (size_t i = 0; i < activity.count; i++) {
                if (activity.inactive_neurons[i] == n) {
                    is_neuron_inactive = 1;
                    break;
                }
            }
            
            if (!is_neuron_inactive) {
                is_channel_active = 1;
            }
        }

        // Jeśli żaden neuron w kanale nie był aktywny, oznacz kanał jako nieużywany
        unused_channels[ch] = !is_channel_active;
    }

    return unused_channels;
}

/**
 * Przeprowadza pruning kanałów wejściowych w warstwie Conv2D
 * @param layer Warstwa Conv2D do modyfikacji
 * @param unused_channels Maska nieużywanych kanałów (1 - nieużywany, 0 - używany)
 */
void prune_conv2d_input_channels(Conv2DLayer* layer, int* unused_channels) {
    int new_in_channels = layer->in_channels - count_unused(unused_channels, layer->in_channels);
    if (new_in_channels <= 0) {
        fprintf(stderr, "Error: Attempting to prune all input channels!\n");
        free(unused_channels);
        return;
    }

    int kernel_size = layer->kernel_size * layer->kernel_size;
    size_t new_weight_size = new_in_channels * layer->out_channels * kernel_size;
    
    // Alokacja nowych wag
    float* new_weights = (float*)malloc(new_weight_size * sizeof(float));
    float* new_weight_grads = NULL;
    if (layer->base.weight_gradients) {
        new_weight_grads = (float*)malloc(new_weight_size * sizeof(float));
    }

    if (!new_weights || (layer->base.weight_gradients && !new_weight_grads)) {
        fprintf(stderr, "Memory allocation failed in prune_conv2d_input_channels\n");
        free(new_weights);
        free(new_weight_grads);
        return;
    }

    // Przepisz tylko aktywne kanały
    int new_ch_idx = 0;
    for (int old_ch_idx = 0; old_ch_idx < layer->in_channels; old_ch_idx++) {
        if (!unused_channels[old_ch_idx]) {
            // Dla każdego filtra przepisz tylko aktywny kanał
            for (int f = 0; f < layer->out_channels; f++) {
                int old_offset = f * (layer->in_channels * kernel_size) + old_ch_idx * kernel_size;
                int new_offset = f * (new_in_channels * kernel_size) + new_ch_idx * kernel_size;
                
                memcpy(&new_weights[new_offset],
                       &layer->base.weights[old_offset],
                       kernel_size * sizeof(float));
                
                if (new_weight_grads) {
                    memcpy(&new_weight_grads[new_offset],
                           &layer->base.weight_gradients[old_offset],
                           kernel_size * sizeof(float));
                }
            }
            new_ch_idx++;
        }
    }

    // Aktualizacja struktury warstwy
    free(layer->base.weights);
    layer->base.weights = new_weights;
    
    if (layer->base.weight_gradients) {
        free(layer->base.weight_gradients);
        layer->base.weight_gradients = new_weight_grads;
    }
    
    layer->in_channels = new_in_channels;
    layer->base.num_inputs = new_in_channels * layer->input_dim * layer->input_dim;
    
    // Aktualizacja wymiarów wyjściowych (nie zmienia się, bo out_channels pozostaje takie samo)
    int output_dim = calculate_output_dim(layer->input_dim, layer->kernel_size, 
                                       layer->stride, layer->padding);
    layer->base.output_size = output_dim * output_dim * layer->out_channels;
    
    // Aktualizacja input_gradients jeśli istnieje
    if (layer->base.input_gradients) {
        free(layer->base.input_gradients);
        layer->base.input_gradients = malloc(layer->base.num_inputs * sizeof(float));
    }
    
    // Logowanie zmian
    printf("Pruned Conv2D input channels: %d -> %d\n", 
           layer->in_channels + count_unused(unused_channels, layer->in_channels), 
           layer->in_channels);
}

