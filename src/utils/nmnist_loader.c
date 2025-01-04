#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include "../../include/utils/nmnist_loader.h"

// Helper function to load a single NMNIST file
NMNISTSample load_nmnist_file(const char *file_path, int label) {
    FILE *file = fopen(file_path, "rb");
    if (!file) {
        printf("Error: Could not open file %s\n", file_path);
        exit(EXIT_FAILURE);
    }

    // Determine the number of events
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    size_t num_events = file_size / 5;  // Each event is 5 bytes (1 byte each for x, y, polarity, 2 bytes for timestamp)

    NMNISTEvent *events = (NMNISTEvent *)malloc(num_events * sizeof(NMNISTEvent));
    if (!events) {
        printf("Error: Memory allocation for NMNIST events failed\n");
        exit(EXIT_FAILURE);
    }

    // Read events from the file
    for (size_t i = 0; i < num_events; i++) {
        unsigned char buffer[5];
        fread(buffer, 1, 5, file);

        events[i].x = buffer[0];
        events[i].y = buffer[1];
        events[i].polarity = buffer[2];
        events[i].timestamp = (buffer[3] << 8) | buffer[4];
    }

    fclose(file);

    // Construct the sample
    NMNISTSample sample;
    sample.num_events = num_events;
    sample.events = events;
    sample.label = label;

    return sample;
}

// Load the entire NMNIST dataset
NMNISTDataset *load_nmnist_dataset(const char *data_dir, size_t max_samples) {
    NMNISTDataset *dataset = (NMNISTDataset *)malloc(sizeof(NMNISTDataset));
    if (!dataset) {
        printf("Error: Memory allocation for NMNIST dataset failed\n");
        exit(EXIT_FAILURE);
    }

    dataset->samples = (NMNISTSample *)malloc(max_samples * sizeof(NMNISTSample));
    if (!dataset->samples) {
        printf("Error: Memory allocation for NMNIST samples failed\n");
        exit(EXIT_FAILURE);
    }

    dataset->num_samples = 0;

    // Iterate through all 10 digit directories (0-9)
    for (int digit = 0; digit < 10; digit++) {
        char digit_dir[256];
        snprintf(digit_dir, sizeof(digit_dir), "%s/%d", data_dir, digit);

        // Open the directory for the digit
        DIR *dir = opendir(digit_dir);
        if (!dir) {
            printf("Error: Could not open directory %s\n", digit_dir);
            exit(EXIT_FAILURE);
        }

        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {
            if (strstr(entry->d_name, ".bin")) {
                // Construct the full file path
                char file_path[512];
                snprintf(file_path, sizeof(file_path), "%s/%s", digit_dir, entry->d_name);

                // Load the NMNIST sample
                NMNISTSample sample = load_nmnist_file(file_path, digit);

                // Add to the dataset
                if (dataset->num_samples < max_samples) {
                    dataset->samples[dataset->num_samples++] = sample;
                } else {
                    break;
                }
            }
        }
        closedir(dir);

        if (dataset->num_samples >= max_samples) {
            break;
        }
    }

    return dataset;
}

// Free the NMNIST dataset
void free_nmnist_dataset(NMNISTDataset *dataset) {
    for (size_t i = 0; i < dataset->num_samples; i++) {
        free(dataset->samples[i].events);
    }
    free(dataset->samples);
    free(dataset);
}
