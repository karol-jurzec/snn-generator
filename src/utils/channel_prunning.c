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
    printf("Resetowanie spike counters i compression...\n");
    
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        
        // Reset compression dla Conv2D
        if (layer->layer_type == LAYER_CONV2D) {
            Conv2DLayer *conv = (Conv2DLayer *)layer;
            
            // 🔄 RESET COMPRESSION - przywróć oryginalne wymiary
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
        
        // Reset spike counters
        if (layer->is_spiking && layer->reset_spike_counts) {
            layer->reset_spike_counts(layer);
            
            SpikingLayer *spiking = (SpikingLayer *)layer;
            if (spiking->total_spikes && spiking->num_neurons > 0) {
                memset(spiking->total_spikes, 0, spiking->num_neurons * sizeof(int));
            }
        }
    }
    
    printf("✅ Reset completed\n");
}

PruningInfo* create_pruning_info(Network *network) {
    PruningInfo *info = (PruningInfo *)malloc(sizeof(PruningInfo));
    info->num_layers = network->num_layers;
    info->inactive_neurons_count = (int *)calloc(network->num_layers, sizeof(int));
    info->inactive_channels = (bool **)malloc(network->num_layers * sizeof(bool *));
    info->pruned_channels_count = (size_t *)calloc(network->num_layers, sizeof(size_t));
    info->sparsity_ratio = 0.0f;
    
    // Inicjalizuj maski dla warstw conv2d
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
    printf("=== Analiza aktywności sieci (threshold: %d) ===\n", threshold);
    
    PruningInfo *info = create_pruning_info(network);
    
    // Przejdź przez wszystkie warstwy
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        
        if (layer->is_spiking) {
            SpikingLayer *spiking = (SpikingLayer *)layer;
            
            // Policz neurony nieaktywne
            int inactive_count = 0;
            for (size_t j = 0; j < spiking->num_neurons; j++) {
                if (spiking->total_spikes[j] <= threshold) {
                    inactive_count++;
                }
            }
            info->inactive_neurons_count[i] = inactive_count;
            
            printf("Warstwa %zu (Spiking): %d/%zu neuronów nieaktywnych\n", 
                   i, inactive_count, spiking->num_neurons);
        }
    }
    
    // Analizuj kanały conv2d
    mark_inactive_channels(network, info, threshold);
    
    return info;
}

void mark_inactive_channels(Network *network, PruningInfo *info, int threshold) {
    printf("\n=== Analiza kanałów konwolucyjnych ===\n");
    
    // pomijamy dwie warstwy ponieważ nie może być dwóch 
    //następujących spikowych po sobie i aby zapobiec out of range index 

    for (size_t i = 0; i < network->num_layers - 2; i++) {  
        LayerBase *conv_layer = network->layers[i];
        LayerBase *next_layer = network->layers[i + 1];
        LayerBase *spiking_layer = network->layers[i + 2];
        
        // Sprawdź pattern: Conv2D → MaxPool2D → SpikingLayer
        if (conv_layer->layer_type == LAYER_CONV2D && 
            next_layer->layer_type == LAYER_MAXPOOL2D && 
            spiking_layer->is_spiking) {
            
            Conv2DLayer *conv = (Conv2DLayer *)conv_layer;
            MaxPool2DLayer *pool = (MaxPool2DLayer *)next_layer;
            SpikingLayer *spiking = (SpikingLayer *)spiking_layer;
            
            printf("Pattern znaleziony: Conv2D[%zu] → MaxPool2D[%zu] → Spiking[%zu]\n", i, i+1, i+2);
            
            // Dla każdego kanału wyjściowego conv2d
            for (int channel = 0; channel < conv->out_channels; channel++) {
                bool channel_inactive = check_channel_inactive(spiking, conv, pool, channel, threshold);
                
                if (channel_inactive) {
                    info->inactive_channels[i][channel] = true;
                    info->pruned_channels_count[i]++;
                    printf("  Kanał %d: NIEAKTYWNY (zostanie usunięty)\n", channel);
                } else {
                    printf("  Kanał %d: aktywny\n", channel);
                }
            }
            
            float pruning_ratio = (float)info->pruned_channels_count[i] / conv->out_channels * 100.0f;
            printf("Warstwa %zu: %zu/%d kanałów zostanie usuniętych (%.1f%%)\n", 
                   i, info->pruned_channels_count[i], conv->out_channels, pruning_ratio);
        }
        
        // Sprawdź czy to para Conv2D -> SpikingLayer
        if (conv_layer->layer_type == LAYER_CONV2D && next_layer->is_spiking) {
            Conv2DLayer *conv = (Conv2DLayer *)conv_layer;
            SpikingLayer *spiking = (SpikingLayer *)next_layer;
            
            printf("Sprawdzam kanały Conv2D w warstwie %zu -> Spiking %zu\n", i, i+1);
            
            // Dla każdego kanału wyjściowego conv2d
            for (int channel = 0; channel < conv->out_channels; channel++) {
                bool channel_inactive = check_channel_inactive(spiking, conv, NULL, channel, threshold);
                
                if (channel_inactive) {
                    info->inactive_channels[i][channel] = true;
                    info->pruned_channels_count[i]++;
                    printf("  Kanał %d: NIEAKTYWNY - zostanie usunięty\n", channel);
                } else {
                    printf("  Kanał %d: aktywny\n", channel);
                }
            }
            
            printf("Warstwa %zu: %zu/%d kanałów zostanie usuniętych\n", 
                   i, info->pruned_channels_count[i], conv->out_channels);

                    printf("🔍 DEBUG: Conv2D[%zu]: %d total channels\n", i, conv->out_channels);
            
            for (int channel = 0; channel < conv->out_channels; channel++) {
                bool channel_inactive = check_channel_inactive(spiking, conv, NULL, channel, threshold);
                
                if (channel_inactive) {
                    info->inactive_channels[i][channel] = true;
                    info->pruned_channels_count[i]++;
                    printf("  ❌ Channel %d: NIEAKTYWNY - zostanie usunięty\n", channel);
                } else {
                    printf("  ✅ Channel %d: aktywny\n", channel);
                }
            }
            
            printf("🎯 RESULT: %zu/%d channels will be pruned (%.1f%%)\n", 
                   info->pruned_channels_count[i], conv->out_channels,
                   (float)info->pruned_channels_count[i] / conv->out_channels * 100.0f);
        }
    }
}

bool check_channel_inactive(SpikingLayer *spiking_layer, Conv2DLayer *conv_layer, 
                                         MaxPool2DLayer *pool_layer, int channel_idx, int threshold)  {
    // Oblicz wymiary wyjściowe conv2d
    int conv_output_h = (conv_layer->input_dim + 2*conv_layer->padding - conv_layer->kernel_size) 
                   / conv_layer->stride + 1;
    int conv_output_w = conv_output_h;  // Zakładamy kwadratowe wyjście

    // jeśli pomiędzy conv2d a spiking layer jest pooling wymiary 
    // wyjściowe nadpisuje wymiarami po poolingu 

    int output_h, output_w; 

    if (pool_layer) {
        output_h = conv_output_h / pool_layer->kernel_size;
        output_w = conv_output_w / pool_layer->kernel_size;
    } else {
        output_h = conv_output_h;
        output_w = conv_output_w;
    }

    int neurons_per_channel = output_h * output_w;
    
    // Sprawdź wszystkie neurony odpowiadające temu kanałowi
    for (int h = 0; h < output_h; h++) {
        for (int w = 0; w < output_w; w++) {
            int neuron_idx = channel_idx * neurons_per_channel + h * output_w + w;
            
            if (neuron_idx < spiking_layer->num_neurons) {
                if (spiking_layer->total_spikes[neuron_idx] > threshold) {
                    return false;  // Znaleziono aktywny neuron w kanale
                }
            }
        }
    }
    
    return true;  
}


void apply_channel_pruning(Network *network, PruningInfo *pruning_info) {
    printf("\n=== Aplikowanie channel pruning z COMPRESSION ===\n");
    
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        
        if (layer->layer_type == LAYER_CONV2D && pruning_info->inactive_channels[i]) {
            Conv2DLayer *conv = (Conv2DLayer *)layer;
            
            if (pruning_info->pruned_channels_count[i] > 0) {
                printf("Warstwa %zu: Applying compression...\n", i);
                
                // 🚀 UŻYWAJ NOWEJ FUNKCJI COMPRESSION (tylko backward)
                apply_channel_compression(conv, 
                                        pruning_info->inactive_channels[i],  // backward mask
                                        NULL);                              // forward mask = NULL
                
                printf("    ✅ Compression applied: %zu out_channels pruned\n", 
                       pruning_info->pruned_channels_count[i]);
            }
        }
    }
    
    printf("✅ Channel pruning with compression applied\n");
}

void print_pruning_stats(PruningInfo *pruning_info) {
    printf("\n=== Statystyki Pruning ===\n");
    
    size_t total_pruned = 0;
    size_t total_channels = 0;
    
    for (size_t i = 0; i < pruning_info->num_layers; i++) {
        if (pruning_info->pruned_channels_count[i] > 0) {
            printf("Warstwa %zu: %zu usuniętych kanałów\n", 
                   i, pruning_info->pruned_channels_count[i]);
            total_pruned += pruning_info->pruned_channels_count[i];
        }
    }
    
    printf("Całkowicie: %zu usuniętych kanałów\n", total_pruned);
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
 //   ""     """"""""   """""       ""  


void test_channel_pruning(const char *architecture_path, const char *weights_path, 
                         const char *dataset_path, int num_samples_for_analysis,
                          int spike_threshold, DatasetFormat format, int input_width, int input_height, 
                        int no_channels) {
    printf("\n==========================================\n");
    printf("ROZPOCZĘCIE TESTU CHANNEL PRUNING\n");
    printf("==========================================\n");
    printf("Architektura: %s\n", architecture_path);
    printf("Wagi: %s\n", weights_path);  
    printf("Dataset: %s\n", dataset_path);
    printf("Próbki do analizy: %d\n", num_samples_for_analysis);
    printf("Threshold spike-ów: %d\n", spike_threshold);
    printf("==========================================\n\n");

    // KROK 1: Wczytaj sieć
    printf("KROK 1: Wczytywanie sieci...\n");
    Network *network = initialize_network_from_file(architecture_path, input_width, input_height, no_channels);
    if (!network) {
        printf("❌ Błąd: Nie udało się wczytać sieci!\n");
        return;
    }
    load_weights_from_json(network, weights_path);
    printf("✅ Sieć wczytana pomyślnie\n\n");

    // KROK 2: Wczytaj dataset i podziel na dwie części
    printf("KROK 2: Wczytywanie i podział datasetu...\n");
    // Load more samples to split into analysis + test sets
    int total_samples = num_samples_for_analysis * 2;  // 250 + 250 = 500
    Dataset *full_dataset = load_dataset(dataset_path, format, total_samples, false, false);
    if (!full_dataset) {
        printf("❌ Błąd: Nie udało się wczytać datasetu!\n");
        if (network) free_network(network);
        return;
    }

    // Create test dataset (first 250 samples: indices 0-249)
    Dataset *test_dataset = (Dataset*)malloc(sizeof(Dataset));
    test_dataset->num_samples = num_samples_for_analysis;
    test_dataset->input_channels = full_dataset->input_channels;
    test_dataset->input_width = full_dataset->input_width;
    test_dataset->input_height = full_dataset->input_height;
    test_dataset->num_classes = full_dataset->num_classes;
    test_dataset->samples = full_dataset->samples;  // Point to first part

    // Create analysis dataset (next 250 samples: indices 250-499)
    Dataset *analysis_dataset = (Dataset*)malloc(sizeof(Dataset));
    analysis_dataset->num_samples = num_samples_for_analysis;
    analysis_dataset->input_channels = full_dataset->input_channels;
    analysis_dataset->input_width = full_dataset->input_width;
    analysis_dataset->input_height = full_dataset->input_height;
    analysis_dataset->num_classes = full_dataset->num_classes;
    analysis_dataset->samples = &full_dataset->samples[num_samples_for_analysis];  // Point to second part

    printf("✅ Dataset podzielony:\n");
    printf("   - Test dataset: próbki 0-%d (%zu próbek)\n", num_samples_for_analysis-1, test_dataset->num_samples);
    printf("   - Analysis dataset: próbki %d-%d (%zu próbek)\n", 
           num_samples_for_analysis, total_samples-1, analysis_dataset->num_samples);
    printf("\n");

    // KROK 3: Test accuracy PRZED pruning (na test dataset)
    printf("KROK 3: Test accuracy PRZED pruning (na test dataset)...\n");
    clock_t start_time = clock();
    float accuracy_before = test(network, test_dataset);
    clock_t end_time = clock();
    double inference_time_before = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("✅ Accuracy przed pruning: %.2f%% (czas: %.3fs)\n\n", accuracy_before, inference_time_before);

    // KROK 4: Reset liczników i analiza aktywności (na analysis dataset)
    printf("KROK 4: Analiza aktywności neuronów (na analysis dataset)...\n");
    reset_spike_counters(network);
    
    // Przepuść próbki do analizy przez sieć
    printf("Przepuszczanie %zu próbek przez sieć do analizy aktywności...\n", analysis_dataset->num_samples);
    
    test(network, analysis_dataset);

    printf("✅ Analiza aktywności zakończona\n\n");

    // KROK 5: Analiza i identyfikacja kanałów do pruning
    printf("KROK 5: Identyfikacja nieaktywnych kanałów (na podstawie analysis dataset)...\n");
    PruningInfo *pruning_info = analyze_network_activity(network, spike_threshold);
    print_pruning_stats(pruning_info);

    // KROK 6: Aplikuj pruning
    printf("\nKROK 6: Aplikowanie pruning...\n");
    apply_channel_pruning(network, pruning_info);
    printf("✅ Pruning zastosowany\n\n");

    // KROK 7: Test accuracy PO pruning (na tym samym test dataset co wcześniej)
    printf("KROK 7: Test accuracy PO pruning (na test dataset)...\n");
    start_time = clock();
    float accuracy_after = test(network, test_dataset);
    end_time = clock();
    double inference_time_after = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("✅ Accuracy po pruning: %.2f%% (czas: %.3fs)\n\n", accuracy_after, inference_time_after);

    // KROK 8: Podsumowanie wyników
    printf("==========================================\n");
    printf("PODSUMOWANIE WYNIKÓW PRUNING\n");
    printf("==========================================\n");
    printf("METODYKA:\n");
    printf("- Analysis dataset:  próbki %d-%d (%zu próbek)\n", 
           num_samples_for_analysis, total_samples-1, analysis_dataset->num_samples);
    printf("- Test dataset:      próbki 0-%d (%zu próbek)\n", 
           num_samples_for_analysis-1, test_dataset->num_samples);
    printf("- Threshold spike-ów: %d\n", spike_threshold);
    printf("\nWYNIKI:\n");
    printf("Accuracy PRZED pruning:    %.2f%%\n", accuracy_before);
    printf("Accuracy PO pruning:       %.2f%%\n", accuracy_after);
    printf("Zmiana accuracy:           %+.2f%% ", accuracy_after - accuracy_before);
    if (accuracy_after - accuracy_before > -1.0f) {
        printf("✅ (akceptowalna)\n");
    } else {
        printf("❌ (znaczący spadek)\n");
    }
    printf("Czas inferencji PRZED:     %.3fs\n", inference_time_before);
    printf("Czas inferencji PO:        %.3fs\n", inference_time_after);
    printf("Przyspieszenie:            %.2fx ", inference_time_before / inference_time_after);
    if (inference_time_after < inference_time_before) {
        printf("✅ (przyspieszenie)\n");
    } else {
        printf("❌ (spowolnienie)\n");
    }
    printf("==========================================\n");

    // Cleanup (be careful not to double-free)
    free_pruning_info(pruning_info);
    
    // Free the wrapper structs but not the underlying samples
    // (they're part of full_dataset)
    free(test_dataset);
    free(analysis_dataset);
    
    // Free the full dataset (this frees the actual samples)
    free_dataset(full_dataset);
    free_network(network);
    
    printf("✅ Test channel pruning zakończony pomyślnie!\n");
}

// Dodaj na koniec istniejącego pliku:
void reset_channel_pruning(Network *network) {
    printf("Resetowanie channel pruning z compression...\n");
    
    // Reset compression i masek dla wszystkich warstw Conv2D
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        
        if (layer->layer_type == LAYER_CONV2D) {
            Conv2DLayer *conv = (Conv2DLayer *)layer;
            
            // 🔄 RESET COMPRESSION ARRAYS
            if (conv->out_active_channels_idx) {
                free(conv->out_active_channels_idx);
                conv->out_active_channels_idx = NULL;
            }
            
            if (conv->in_active_channels_idx) {
                free(conv->in_active_channels_idx);
                conv->in_active_channels_idx = NULL;
            }
            
            // 🔄 PRZYWRÓĆ ORYGINALNE WYMIARY
            conv->out_channels = conv->original_out_channels;
            conv->in_channels = conv->original_in_channels;
            
            printf("  Conv2D[%zu]: Reset - back to %d out_channels, %d in_channels\n", 
                   i, conv->out_channels, conv->in_channels);
        }
    }
    
    printf("✅ Channel pruning reset completed\n");
}

typedef struct {
    int threshold;
    size_t total_pruned_channels;
    float pruning_percentage;
    float accuracy_before;
    float accuracy_after;
    float accuracy_drop;
    double inference_time_before;
    double inference_time_after;
    float speedup;
    size_t total_channels;
} ThresholdResult;

void generate_threshold_array(int **thresholds, int *num_thresholds) {
    // Alokuj tablicę (maksymalnie ~400 wartości dla dobrego coverage krytycznego zakresu)
    int capacity = 400;
    *thresholds = (int*)malloc(capacity * sizeof(int));
    int count = 0;
    
    // Rzadkie próbkowanie dla niskich wartości (0-200): co 10 (nie ma co testować gęsto)
    for (int t = 0; t <= 200 && count < capacity; t += 10) {
        (*thresholds)[count++] = t;
    }
    
    // GĘSTE próbkowanie dla krytycznego zakresu (201-350): co 1 (!!)
    // Tu się dzieje najwięcej akcji zgodnie z Twoimi obserwacjami
    for (int t = 201; t <= 350 && count < capacity; t += 1) {
        (*thresholds)[count++] = t;
    }
    
    // Średnie próbkowanie dla zakresu spadku (351-600): co 3
    for (int t = 353; t <= 600 && count < capacity; t += 3) {
        (*thresholds)[count++] = t;
    }
    
    // Rzadsze próbkowanie (601-1000): co 5
    for (int t = 605; t <= 1000 && count < capacity; t += 5) {
        (*thresholds)[count++] = t;
    }
    
    // Bardzo rzadkie (1001-2000): co 25
    for (int t = 1025; t <= 2000 && count < capacity; t += 25) {
        (*thresholds)[count++] = t;
    }
    
    // Najrzadsze dla największych (2001-4000): co 100
    for (int t = 2100; t <= 4000 && count < capacity; t += 100) {
        (*thresholds)[count++] = t;
    }
    
    *num_thresholds = count;
    
    printf("Wygenerowano %d wartości threshold (0-%d)\n", count, (*thresholds)[count-1]);
    printf("Szczególne skupienie na krytycznym zakresie 201-350 (co 1)\n");
}

void study_threshold_impact(const char *architecture_path, const char *weights_path, 
                           const char *dataset_path, const char *results_file, 
                           int num_samples_for_analysis) {
    
    // Wygeneruj tablicę thresholds dynamicznie
    int *thresholds;
    int num_thresholds;
    generate_threshold_array(&thresholds, &num_thresholds);
    
    // ZMIANA: Ustaw różne rozmiary dla różnych celów
    int num_samples_for_testing = 50;  // Tylko 50 próbek do pomiaru accuracy (szybciej!)
    
    printf("\n==============================================\n");
    printf("STUDIUM WPŁYWU THRESHOLD NA PRUNING (ROZSZERZONE)\n");
    printf("==============================================\n");
    printf("Architektura: %s\n", architecture_path);
    printf("Wagi: %s\n", weights_path);
    printf("Dataset: %s\n", dataset_path);
    printf("Próbki do analizy: %d\n", num_samples_for_analysis);
    printf("Próbki do testów: %d (zoptymalizowane!)\n", num_samples_for_testing);  // ZMIANA
    printf("Plik wyników: %s\n", results_file);
    printf("Liczba threshold values: %d (0 do %d)\n", num_thresholds, thresholds[num_thresholds-1]);
    printf("==============================================\n\n");

    // KROK 1: Wczytaj sieć (raz na początku)
    printf("KROK 1: Wczytywanie sieci...\n");
    Network *network = initialize_network_from_file(architecture_path, 10, 10, 2);
    if (!network) {
        printf("❌ Błąd: Nie udało się wczytać sieci!\n");
        free(thresholds);
        return;
    }
    load_weights_from_json(network, weights_path);
    printf("✅ Sieć wczytana pomyślnie\n\n");

    // KROK 2: Wczytaj i podziel dataset 
    printf("KROK 2: Wczytywanie datasetu...\n");
    // ZMIANA: 50 do testów + num_samples_for_analysis do analizy
    int total_samples = num_samples_for_testing + num_samples_for_analysis;  // 50 + 250 = 300
    Dataset *full_dataset = load_dataset(dataset_path, FORMAT_STMNIST, total_samples, false, false);
    if (!full_dataset) {
        printf("❌ Błąd: Nie udało się wczytać datasetu!\n");
        free_network(network);
        free(thresholds);
        return;
    }

    // ZMIANA: Test dataset - tylko pierwsze 50 próbek
    Dataset *test_dataset = (Dataset*)malloc(sizeof(Dataset));
    test_dataset->num_samples = num_samples_for_testing;  // 50
    test_dataset->input_channels = full_dataset->input_channels;
    test_dataset->input_width = full_dataset->input_width;
    test_dataset->input_height = full_dataset->input_height;
    test_dataset->num_classes = full_dataset->num_classes;
    test_dataset->samples = full_dataset->samples;  // próbki 0-49

    // ZMIANA: Analysis dataset - próbki 50-299 (250 próbek)
    Dataset *analysis_dataset = (Dataset*)malloc(sizeof(Dataset));
    analysis_dataset->num_samples = num_samples_for_analysis;  // 250
    analysis_dataset->input_channels = full_dataset->input_channels;
    analysis_dataset->input_width = full_dataset->input_width;
    analysis_dataset->input_height = full_dataset->input_height;
    analysis_dataset->num_classes = full_dataset->num_classes;
    analysis_dataset->samples = &full_dataset->samples[num_samples_for_testing];  // próbki 50-299

    printf("✅ Dataset przygotowany\n");
    printf("   - Test dataset: próbki 0-%d (%d próbek) - szybki pomiar accuracy\n", 
           num_samples_for_testing-1, num_samples_for_testing);
    printf("   - Analysis dataset: próbki %d-%d (%d próbek) - analiza aktywności\n", 
           num_samples_for_testing, total_samples-1, num_samples_for_analysis);
    printf("\n");

    // Reszta kodu pozostaje bez zmian...
    // KROK 3: Otwórz plik do zapisania wyników
    FILE *results_csv = fopen(results_file, "w");
    if (!results_csv) {
        printf("❌ Błąd: Nie udało się otworzyć pliku wyników!\n");
        free(test_dataset);
        free(analysis_dataset);
        free_dataset(full_dataset);
        free_network(network);
        free(thresholds);
        return;
    }

    // Zapisz nagłówek CSV z informacją o rozmiarach próbek
    fprintf(results_csv, "# Test samples: %d, Analysis samples: %d\n", num_samples_for_testing, num_samples_for_analysis);
    fprintf(results_csv, "threshold,total_channels,pruned_channels,pruning_percentage,accuracy_before,accuracy_after,accuracy_drop,inference_time_before,inference_time_after,speedup,efficiency_score\n");
    printf("✅ Plik wyników przygotowany\n\n");

    // Policz łączną liczbę kanałów w sieci
    size_t total_network_channels = 0;
    for (size_t i = 0; i < network->num_layers; i++) {
        LayerBase *layer = network->layers[i];
        if (layer->layer_type == LAYER_CONV2D) {
            Conv2DLayer *conv = (Conv2DLayer *)layer;
            total_network_channels += conv->out_channels;
        }
    }

    // Zmierz baseline accuracy (bez pruning) na małej próbce
    printf("KROK 3: Pomiar baseline accuracy (na %d próbkach)...\n", num_samples_for_testing);
    reset_channel_pruning(network);
    clock_t start_time = clock();
    float baseline_accuracy = test(network, test_dataset);  // Tylko 50 próbek!
    clock_t end_time = clock();
    double baseline_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("✅ Baseline accuracy: %.2f%% (czas: %.3fs, %d próbek)\n\n", baseline_accuracy, baseline_time, num_samples_for_testing);

    // KROK 4: Iteruj przez threshold values z progress indicator
    printf("KROK 4: Rozpoczęcie testowania %d thresholds (szybko dzięki małej próbce testowej)...\n\n", num_thresholds);
    
    // Reszta pętli pozostaje identyczna - tylko testy accuracy będą znacznie szybsze!
    for (int t_idx = 0; t_idx < num_thresholds; t_idx++) {
        int current_threshold = thresholds[t_idx];
        
        // Progress indicator
        if (t_idx % 20 == 0) {
            printf("Progress: %d/%d (%.1f%%) - obecnie threshold=%d\n", 
                   t_idx, num_thresholds, (float)t_idx/num_thresholds*100, current_threshold);
        }
        
        // Reset pruning
        reset_channel_pruning(network);
        
        // Analiza aktywności na analysis dataset (250 próbek)
        reset_spike_counters(network);
        
        // Przepuść analysis samples
        for (size_t i = 0; i < analysis_dataset->num_samples; i++) {
            Sample *sample = &analysis_dataset->samples[i];
            size_t input_size_per_bin = analysis_dataset->input_channels * 
                                       analysis_dataset->input_width * 
                                       analysis_dataset->input_height;

            for (size_t l = 0; l < network->num_layers; l++) {
                if (network->layers[l]->is_spiking) {
                    network->layers[l]->reset_spike_counts(network->layers[l]);
                }
            }

            for (int t = 0; t < sample->num_bins; t++) {
                float *frame = &sample->input[t * input_size_per_bin];
                network->layers[0]->forward(network->layers[0], frame, input_size_per_bin, 0);
                for (size_t j = 1; j < network->num_layers; j++) {
                    network->layers[j]->forward(network->layers[j], 
                                              network->layers[j-1]->output,
                                              network->layers[j-1]->output_size, 0);
                }
            }

            for (size_t l = 0; l < network->num_layers; l++) {
                if (network->layers[l]->is_spiking) {
                    SpikingLayer *spiking = (SpikingLayer *)network->layers[l];
                    for (size_t n = 0; n < spiking->num_neurons; n++) {
                        LIFNeuron *neuron = (LIFNeuron *)spiking->neurons[n];
                        spiking->total_spikes[n] += neuron->spike_count;
                    }
                }
            }
        }
        
        // Aplikuj pruning (szybkie, bez printów)
        PruningInfo *pruning_info = create_pruning_info(network);
        
        for (size_t i = 0; i < network->num_layers - 1; i++) {
            LayerBase *layer = network->layers[i];
            LayerBase *next_layer = network->layers[i + 1];
            
            if (layer->layer_type == LAYER_CONV2D && next_layer->is_spiking) {
                Conv2DLayer *conv = (Conv2DLayer *)layer;
                SpikingLayer *spiking = (SpikingLayer *)next_layer;
                
                for (int channel = 0; channel < conv->out_channels; channel++) {
                    bool channel_inactive = check_channel_inactive(spiking, conv, NULL, channel, current_threshold);
                    if (channel_inactive) {
                        pruning_info->inactive_channels[i][channel] = true;
                        pruning_info->pruned_channels_count[i]++;
                    }
                }
            }
        }
        
        apply_channel_pruning(network, pruning_info);
        
        // Policz statystyki pruning
        size_t total_pruned = 0;
        for (size_t i = 0; i < pruning_info->num_layers; i++) {
            total_pruned += pruning_info->pruned_channels_count[i];
        }
        float pruning_percentage = (float)total_pruned / total_network_channels * 100.0f;
        
        // Zmierz accuracy po pruning (tylko 50 próbek - bardzo szybko!)
        start_time = clock();
        float accuracy_after = test(network, test_dataset);  // 50 próbek
        end_time = clock();
        double inference_time_after = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        
        float accuracy_drop = baseline_accuracy - accuracy_after;
        float speedup = (float)(baseline_time / inference_time_after);
        
        // Oblicz efficiency score
        float efficiency_score = 0.0f;
        if (accuracy_drop > 0.0f) {
            efficiency_score = fmin(speedup / accuracy_drop, 100.0f);
        } else if (speedup > 1.0f) {
            efficiency_score = 100.0f;
        }
        
        // Zapisz do pliku CSV
        fprintf(results_csv, "%d,%zu,%zu,%.2f,%.2f,%.2f,%.2f,%.6f,%.6f,%.2f,%.2f\n",
                current_threshold,
                total_network_channels,
                total_pruned,
                pruning_percentage,
                baseline_accuracy,
                accuracy_after,
                accuracy_drop,
                baseline_time,
                inference_time_after,
                speedup,
                efficiency_score);
        fflush(results_csv);
        
        // Pokaż wyniki dla wybranych thresholds
        if (t_idx % 20 == 0 || current_threshold == 0 || current_threshold <= 10) {
            printf("  Threshold %4d: %5.1f%% pruned, acc %.2f%% → %.2f%% (%.2f%% drop), %.2fx speedup\n",
                   current_threshold, pruning_percentage, baseline_accuracy, accuracy_after, 
                   accuracy_drop, speedup);
        }
        
        free_pruning_info(pruning_info);
    }
    
    printf("\n==============================================\n");
    printf("ZAKOŃCZONO! Przetestowano %d thresholds (0-%d)\n", num_thresholds, thresholds[num_thresholds-1]);
    printf("Szybkość dzięki małej próbce testowej: %d próbek\n", num_samples_for_testing);
    printf("==============================================\n");
    
    // Cleanup
    fclose(results_csv);
    free(thresholds);
    free(test_dataset);
    free(analysis_dataset);
    free_dataset(full_dataset);
    free_network(network);
    
    printf("✅ Rozszerzone studium threshold zakończone!\n");
    printf("   Wyniki zapisane w: %s\n", results_file);
    printf("   Metoda: 250 próbek do analizy + 50 próbek do pomiaru accuracy\n");
}