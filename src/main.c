#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../include/network.h"
#include "../include/utils/network_loader.h"
#include "../include/utils/dataset_loader.h"
#include "../include/utils/bidirectional_pruning.h"

#define NUM_SAMPLES_SCENARIO 1000

// Simple progress bar
void print_progress(int current, int total) {
    int width = 50;
    float ratio = (float)current / total;
    int pos = (int)(width * ratio);

    printf("[");
    for (int i = 0; i < width; i++) {
        if (i < pos) printf("=");
        else if (i == pos) printf(">");
        else printf(" ");
    }
    printf("] %3d%%\r", (int)(ratio * 100));
    fflush(stdout);
}

// Compute F1 score
void compute_f1(int *preds, int *labels, int n, float *f1, float *accuracy) {
    int tp[10] = {0}, fp[10] = {0}, fn[10] = {0};
    int correct = 0;

    for (int i = 0; i < n; i++) {
        if (preds[i] == labels[i]) correct++;
        for (int c = 0; c < 10; c++) {
            if (preds[i] == c && labels[i] == c) tp[c]++;
            if (preds[i] == c && labels[i] != c) fp[c]++;
            if (preds[i] != c && labels[i] == c) fn[c]++;
        }
    }

    *accuracy = (float)correct / n * 100.0f;

    float f1_sum = 0.0f;
    for (int c = 0; c < 10; c++) {
        float p = tp[c] / (float)(tp[c] + fp[c] + 1e-8f);
        float r = tp[c] / (float)(tp[c] + fn[c] + 1e-8f);
        float f1c = 2 * p * r / (p + r + 1e-8f);
        f1_sum += f1c;
    }
    *f1 = f1_sum / 10.0f;
}

// Run inference on a dataset
void run_inference(Network *network, Dataset *dataset, int num_samples,
                   int *predictions, double *avg_time) {
    clock_t start = clock();

    for (int i = 0; i < num_samples; i++) {
        print_progress(i, num_samples);

        Sample *sample = &dataset->samples[i];
        predictions[i] = predict_single_sample(network, sample, dataset);
    }
    print_progress(num_samples, num_samples);
    printf("\n");

    clock_t end = clock();
    *avg_time = ((double)(end - start)) / CLOCKS_PER_SEC / num_samples;
}

double compute_avg_time_steps(Dataset *dataset, int num_samples) {
    size_t total_bins = 0;
    for (int i = 0; i < num_samples; i++) {
        total_bins += dataset->samples[i].num_bins;
    }
    return (double)total_bins / num_samples;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s <architecture.json> <weights.json> <dataset_folder>\n", argv[0]);
        return 1;
    }

    const char *arch_path = argv[1];
    const char *weights_path = argv[2];
    const char *dataset_path = argv[3];

    // --- Load network ---
    printf("Loading network...\n");
    Network *network = initialize_network_from_file(arch_path, 34, 34, 2);
    load_weights_from_json(network, weights_path);
    printf("Network loaded.\n");

    // --- Load dataset ---
    printf("Loading dataset...\n");
    Dataset *dataset = load_dataset(dataset_path, FORMAT_NMNIST, NUM_SAMPLES_SCENARIO * 3, false, false);
    if (!dataset) {
        fprintf(stderr, "Failed to load dataset.\n");
        free_network(network);
        return 1;
    }
    printf("Dataset loaded (%zu samples).\n", dataset->num_samples);

    // --- Split samples ---
    Dataset *val_set = create_empty_dataset(NUM_SAMPLES_SCENARIO, 34, 34, 2, 10, 0, 0);
    Dataset *test_set_no_prune = create_empty_dataset(NUM_SAMPLES_SCENARIO, 34, 34, 2, 10, 0, 0);
    Dataset *test_set_prune = create_empty_dataset(NUM_SAMPLES_SCENARIO, 34, 34, 2, 10, 0, 0);

    for (int i = 0; i < NUM_SAMPLES_SCENARIO; i++) {
        val_set->samples[i] = dataset->samples[i];
        test_set_no_prune->samples[i] = dataset->samples[NUM_SAMPLES_SCENARIO + i];
        test_set_prune->samples[i] = dataset->samples[NUM_SAMPLES_SCENARIO * 2 + i];
    }

    // --- SCENARIO 1: NO PRUNING ---
    printf("\n=== Scenario 1: No pruning ===\n");
    int preds_no_prune[NUM_SAMPLES_SCENARIO];
    double avg_time_no_prune;

    clock_t start_no = clock();
    run_inference(network, test_set_no_prune, NUM_SAMPLES_SCENARIO, preds_no_prune, &avg_time_no_prune);
    clock_t end_no = clock();
    double total_time_no = ((double)(end_no - start_no)) / CLOCKS_PER_SEC;

    int labels_no_prune[NUM_SAMPLES_SCENARIO];
    for (int i = 0; i < NUM_SAMPLES_SCENARIO; i++)
        labels_no_prune[i] = test_set_no_prune->samples[i].label;

    float f1_no_prune, acc_no_prune;
    compute_f1(preds_no_prune, labels_no_prune, NUM_SAMPLES_SCENARIO, &f1_no_prune, &acc_no_prune);

    double avg_time_steps_no = compute_avg_time_steps(test_set_no_prune, NUM_SAMPLES_SCENARIO);

    printf("No Pruning -> Accuracy: %.2f%%, F1: %.3f, Avg Time: %.6f s/sample, Total Time: %.3fs, Avg Time Steps: %.1f\n",
           acc_no_prune, f1_no_prune, avg_time_no_prune, total_time_no, avg_time_steps_no);

    // --- SCENARIO 2: WITH BIDIRECTIONAL PRUNING ---
    printf("\n=== Scenario 2: With pruning ===\n");

    int spike_threshold = 0;
    BidirectionalPruningInfo *prune_info = analyze_bidirectional_activity(network, spike_threshold);
    apply_bidirectional_pruning(network, prune_info);
    print_bidirectional_stats(prune_info);

    int preds_prune[NUM_SAMPLES_SCENARIO];
    double avg_time_prune;

    clock_t start_prune = clock();
    run_inference(network, test_set_prune, NUM_SAMPLES_SCENARIO, preds_prune, &avg_time_prune);
    clock_t end_prune = clock();
    double total_time_prune = ((double)(end_prune - start_prune)) / CLOCKS_PER_SEC;

    int labels_prune[NUM_SAMPLES_SCENARIO];
    for (int i = 0; i < NUM_SAMPLES_SCENARIO; i++)
        labels_prune[i] = test_set_prune->samples[i].label;

    float f1_prune, acc_prune;
    compute_f1(preds_prune, labels_prune, NUM_SAMPLES_SCENARIO, &f1_prune, &acc_prune);

    double avg_time_steps_prune = compute_avg_time_steps(test_set_prune, NUM_SAMPLES_SCENARIO);

    printf("Pruned -> Accuracy: %.2f%%, F1: %.3f, Avg Time: %.6f s/sample, Total Time: %.3fs, Avg Time Steps: %.1f\n",
           acc_prune, f1_prune, avg_time_prune, total_time_prune, avg_time_steps_prune);

    // --- Speed-up calculation ---
    if (total_time_prune > 0.0) {
        double speedup = total_time_no / total_time_prune;
        printf("Overall speed-up: %.2fx faster with pruning\n", speedup);
    }

    // --- Cleanup ---
    free_bidirectional_pruning_info(prune_info);
    free_dataset(dataset);
    free_dataset(val_set);
    free_dataset(test_set_no_prune);
    free_dataset(test_set_prune);
    free_network(network);

    return 0;
}