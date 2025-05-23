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
        if (layer->neuron_layer.total_spike_counts[i] == 0) {
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
    int new_in_channels = conv_layer->in_channels - count_unused(unused_channels, size);
    
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

    int kernel_elems = layer->kernel_size * layer->kernel_size;
    size_t new_weight_size = layer->in_channels * new_out_channels * kernel_elems;
    
    // 1. Alokacja nowych wag i biasów
    float* new_weights = malloc(new_weight_size * sizeof(float));
    float* new_biases = malloc(new_out_channels * sizeof(float));
    float* new_weight_grads = malloc(new_weight_size * sizeof(float));
    
    // 2. Przepisywanie tylko aktywnych filtrów
    int new_f_idx = 0;
    for (int old_f_idx = 0; old_f_idx < layer->out_channels; old_f_idx++) {
        if (!unused_filters[old_f_idx]) {
            // Kopiuj wagi dla całego filtra
            memcpy(&new_weights[new_f_idx * layer->in_channels * kernel_elems],
                   &layer->base.weights[old_f_idx * layer->in_channels * kernel_elems],
                   layer->in_channels * kernel_elems * sizeof(float));
            
            // Kopiuj bias i gradienty
            new_biases[new_f_idx] = layer->biases[old_f_idx];
            
            // if (layer->base.weight_gradients) {
            //     memcpy(&new_weight_grads[new_f_idx * layer->in_channels * kernel_elems],
            //            &layer->base.weight_gradients[old_f_idx * layer->in_channels * kernel_elems],
            //            layer->in_channels * kernel_elems * sizeof(float));
            // }
            
            new_f_idx++;
        }
    }
    
    // 3. Aktualizacja struktury warstwy
    free(layer->base.weights);
    free(layer->biases);
    // if (layer->base.weight_gradients) free(layer->base.weight_gradients);
    
    layer->base.weights = new_weights;
    layer->biases = new_biases;
    // layer->base.weight_gradients = new_weight_grads;
    layer->out_channels = new_out_channels;
    
    // 4. Aktualizacja wymiarów wyjściowych
    int output_dim = calculate_output_dim(layer->input_dim, layer->kernel_size, 
                                       layer->stride, layer->padding);
    layer->base.output_size = output_dim * output_dim * new_out_channels;
    
    // 5. Realokacja output buffer jeśli potrzebne
    if (layer->base.output_size != output_dim * output_dim * layer->out_channels) {
        free(layer->base.output);
        layer->base.output = malloc(layer->base.output_size * sizeof(float));
    }
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

int* find_unused_input_channels(Conv2DLayer* layer, NeuronActivity activity) {
    // Dla warstwy Conv2D, kanały wejściowe odpowiadają aktywności neuronów w poprzedniej warstwie spikingowej
    int* unused_channels = calloc(layer->in_channels, sizeof(int));

    // 2 output channels 2 x 2
    // total_spike_counts=[0, 0, 0, 0, 35, 35, 0, 2]
    // inactive_neurons=[0, 1, 2, 3, 6]
    
    // Zakładamy, że activity zawiera informację o nieaktywnych neuronach z poprzedniej warstwy
    for (size_t i = 0; i < activity.count; i++) {
        int neuron_idx = activity.inactive_neurons[i];
        if (neuron_idx < layer->in_channels) {
            unused_channels[neuron_idx] = 1;
        }
    }
    
    return unused_channels;
}

void prune_conv2d_input_channels(Conv2DLayer* layer, int* unused_channels) {
    int new_in_channels = layer->in_channels - count_unused(unused_channels, layer->in_channels);
    int kernel_elems = layer->kernel_size * layer->kernel_size;
    size_t new_weight_size = new_in_channels * layer->out_channels * kernel_elems;
    
    // 1. Alokacja nowych wag
    float* new_weights = malloc(new_weight_size * sizeof(float));
    
    // 2. Przepisywanie tylko aktywnych kanałów
    int new_ch_idx = 0;
    for (int old_ch_idx = 0; old_ch_idx < layer->in_channels; old_ch_idx++) {
        if (!unused_channels[old_ch_idx]) {
            // Dla każdego filtra przepisz tylko aktywny kanał
            for (int f = 0; f < layer->out_channels; f++) {
                int old_offset = f * (layer->in_channels * kernel_elems) + old_ch_idx * kernel_elems;
                int new_offset = f * (new_in_channels * kernel_elems) + new_ch_idx * kernel_elems;
                
                memcpy(&new_weights[new_offset],
                       &layer->base.weights[old_offset],
                       kernel_elems * sizeof(float));
            }
            new_ch_idx++;
        }
    }
    
    // 3. Aktualizacja struktury warstwy
    free(layer->base.weights);
    layer->base.weights = new_weights;
    layer->in_channels = new_in_channels;
    layer->base.num_inputs = new_in_channels * layer->input_dim * layer->input_dim;
    
    // 4. Aktualizacja wymiarów wyjściowych
    layer->base.output_size = calculate_output_dim(layer->input_dim, layer->kernel_size, 
                                                layer->stride, layer->padding) *
                            calculate_output_dim(layer->input_dim, layer->kernel_size, 
                                                layer->stride, layer->padding) *
                            layer->out_channels;
}
