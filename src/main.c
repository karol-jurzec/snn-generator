#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/network.h"
#include "../include/utils/network_loader.h"
#include "../include/utils/dataset_loader.h"

void print_usage(const char *program_name) {
    printf("Usage: %s <architecture.json> <weights.json> <sample_file> [options]\n", program_name);
    printf("\n");
    printf("Arguments:\n");
    printf("  architecture.json    Path to architecture JSON file\n");
    printf("  weights.json        Path to weights JSON file\n");
    printf("  sample_file         Path to sample file (.bin for NMNIST, .mat for STMNIST)\n");
    printf("\n");
    printf("Options:\n");
    printf("  -W, --width WIDTH      Input width (default: auto-detect from format)\n");
    printf("  -H, --height HEIGHT    Input height (default: auto-detect from format)\n");
    printf("  -c, --channels CH      Input channels (default: 2)\n");
    printf("  -h, --help             Show this help message\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s scnn_stmnist_architecture.json scnn_stmnist_weights_bs_64.json sample.mat 5\n", program_name);
    printf("  %s snn_nmnist_architecture.json snn_nmnist_weights_bs_32.json sample.bin 3\n", program_name);
    printf("\n");
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const char *architecture_path = argv[1];
    const char *weights_path = argv[2];
    const char *sample_file = argv[3];
    int label = -1;  // -1 means label not provided
    int input_width = 0;   // 0 means auto-detect
    int input_height = 0;
    int input_channels = 2;

    if (argc >= 5) {
        label = atoi(argv[4]);
        if (label < 0 || label > 9) {
            fprintf(stderr, "Warning: Label %d is out of range (0-9), ignoring\n", label);
            label = -1;
        }
    }

    for (int i = 5; i < argc; i++) {
        if (strcmp(argv[i], "-W") == 0 || strcmp(argv[i], "--width") == 0) {
            if (i + 1 < argc) {
                input_width = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "-H") == 0 || strcmp(argv[i], "--height") == 0) {
            if (i + 1 < argc) {
                input_height = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--channels") == 0) {
            if (i + 1 < argc) {
                input_channels = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    printf("=== SpikeEdge Runtime - Single Sample Prediction ===\n\n");
    printf("Architecture: %s\n", architecture_path);
    printf("Weights: %s\n", weights_path);
    printf("Sample file: %s\n", sample_file);
    if (label >= 0) {
        printf("Label: %d\n", label);
    }
    printf("\n");

    DatasetFormat format = detect_format_from_file(sample_file);
    if (format == FORMAT_UNKNOWN) {
        fprintf(stderr, "Error: Could not detect format from file extension. Supported: .bin (NMNIST), .mat (STMNIST)\n");
        return 1;
    }

    if (input_width == 0 || input_height == 0) {
        if (format == FORMAT_NMNIST) {
            input_width = 34;
            input_height = 34;
        } else if (format == FORMAT_STMNIST) {
            input_width = 10;
            input_height = 10;
        }
    }

    printf("Format: %s\n", (format == FORMAT_NMNIST) ? "NMNIST" : "STMNIST");
    printf("Input dimensions: %dx%dx%d\n\n", input_width, input_height, input_channels);

    printf("Loading network...\n");
    Network *network = initialize_network_from_file(architecture_path, input_width, input_height, input_channels);
    if (!network) {
        fprintf(stderr, "Error: Failed to load network from %s\n", architecture_path);
        return 1;
    }

    printf("Loading weights...\n");
    load_weights_from_json(network, weights_path);
    printf("Network loaded successfully.\n\n");

    printf("Loading sample from %s...\n", sample_file);
    Dataset *dataset = load_single_sample_file(sample_file, label, FORMAT_UNKNOWN, false, false);
    if (!dataset || dataset->num_samples == 0) {
        fprintf(stderr, "Error: Failed to load sample from %s\n", sample_file);
        free_network(network);
        return 1;
    }

    printf("Sample loaded successfully.\n\n");

    Sample *sample = &dataset->samples[0];
    printf("Processing sample...\n");
    if (label >= 0 && sample->label != label) {
        printf("Warning: Dataset label (%d) differs from provided label (%d)\n", sample->label, label);
        printf("Using dataset label: %d\n", sample->label);
    }

    int predicted_label = predict_single_sample(network, sample, dataset);

    printf("\n=== Prediction Results ===\n");
    printf("Predicted label: %d\n", predicted_label);
    if (label >= 0 || sample->label >= 0) {
        printf("Correct: %s\n", (predicted_label == sample->label) ? "YES" : "NO");
    }
    printf("\n");

    free_dataset(dataset);
    free_network(network);

    return (predicted_label == sample->label) ? 0 : 1;
}