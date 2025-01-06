#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <dirent.h>
#include "../../include/utils/nmnist_loader.h"

// Helper to stabilize the dataset (motion compensation)
static void stabilize_events(NMNISTEvent *events, size_t num_events) {
    for (size_t i = 0; i < num_events; i++) {
        unsigned int ts = events[i].timestamp;

        if (ts <= 105000) {
            events[i].x -= (3.5f * ts / 105000);
            events[i].y -= (7.0f * ts / 105000);
        } else if (ts <= 210000) {
            events[i].x -= (3.5f + (3.5f * (ts - 105000) / 105000));
            events[i].y -= (7.0f - (7.0f * (ts - 105000) / 105000));
        } else {
            events[i].x -= (7.0f - (7.0f * (ts - 210000) / 105000));
        }

        // Clamp coordinates and polarity
        if (events[i].x < 1) events[i].x = 1;
        if (events[i].y < 1) events[i].y = 1;
    }
}

// Helper function to load a single NMNIST file
static NMNISTSample load_nmnist_file(const char *file_path, int label, bool stabilize) {
    FILE *file = fopen(file_path, "rb");
    if (!file) {
        printf("Error: Could not open file %s\n", file_path);
        exit(EXIT_FAILURE);
    }

    // Determine the number of events
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    size_t num_events = file_size / 5;  // Each event is 5 bytes
    NMNISTEvent *events = (NMNISTEvent *)malloc(num_events * sizeof(NMNISTEvent));
    if (!events) {
        printf("Error: Memory allocation for NMNIST events failed\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    // Read and decode events
    for (size_t i = 0; i < num_events; i++) {
        unsigned char buffer[5];
        fread(buffer, 1, 5, file);

        events[i].x = buffer[0];  // X-address
        events[i].y = buffer[1];  // Y-address
        events[i].polarity = (buffer[2] >> 7) & 1;  // Polarity (MSB)
        events[i].timestamp = ((buffer[2] & 0x7F) << 16) | (buffer[3] << 8) | buffer[4];  // Timestamp
    }

    fclose(file);

    // Stabilize events if required
    if (stabilize) {
        stabilize_events(events, num_events);
    }

    // Construct the sample
    NMNISTSample sample;
    sample.num_events = num_events;
    sample.events = events;
    sample.label = label;

    return sample;
}

// Load the entire NMNIST dataset
NMNISTDataset *load_nmnist_dataset(const char *data_dir, size_t max_samples, bool stabilize) {
    NMNISTDataset *dataset = (NMNISTDataset *)malloc(sizeof(NMNISTDataset));
    if (!dataset) {
        printf("Error: Memory allocation for NMNIST dataset failed\n");
        exit(EXIT_FAILURE);
    }

    dataset->samples = (NMNISTSample *)malloc(max_samples * sizeof(NMNISTSample));
    if (!dataset->samples) {
        printf("Error: Memory allocation for NMNIST samples failed\n");
        free(dataset);
        exit(EXIT_FAILURE);
    }

    dataset->num_samples = 0;

    // Iterate through all 10 digit directories (0-9)
    for (int digit = 0; digit < 10; digit++) {
        char digit_dir[256];
        snprintf(digit_dir, sizeof(digit_dir), "%s/%d", data_dir, digit);

        DIR *dir = opendir(digit_dir);
        if (!dir) {
            printf("Error: Could not open directory %s\n", digit_dir);
            continue;  // Skip invalid directories
        }

        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {
            if (strstr(entry->d_name, ".bin")) {
                // Construct the full file path
                char file_path[512];
                snprintf(file_path, sizeof(file_path), "%s/%s", digit_dir, entry->d_name);

                // Load the NMNIST sample
                NMNISTSample sample = load_nmnist_file(file_path, digit, stabilize);

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

// Function to convert NMNIST events to discretized input
float *convert_events_to_input(const NMNISTEvent *events, size_t num_events, int time_bins, int height, int width, unsigned int max_time) {
    // Allocate a 3D array: [T][H][W]
    size_t input_size = time_bins * height * width;
    float *input = (float *)calloc(input_size, sizeof(float)); // Initialize to 0

    unsigned int bin_size = max_time / time_bins;

    // Populate the bins
    for (size_t i = 0; i < num_events; i++) {
        const NMNISTEvent *event = &events[i];

        // Determine the time bin
        int t = event->timestamp / bin_size;
        if (t >= time_bins) continue; // Ignore events outside the time range

        // Map (x, y) to the spatial dimensions
        int x = event->x - 1; // Convert 1-based to 0-based indexing
        int y = event->y - 1;
        if (x < 0 || x >= width || y < 0 || y >= height) continue; // Ignore invalid coordinates

        // Compute the flattened index
        size_t index = (t * height * width) + (y * width) + x;

        // Accumulate polarity (1 for ON spikes, -1 for OFF spikes)
        input[index] += (event->polarity == 1) ? 1.0f : 0.0f;
    }

    return input; // Caller must free the memory
}

// Save a single 2D frame as a PGM image
void save_frame_as_pgm(const char *filename, float *data, int height, int width) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Error: Could not open file %s for writing\n", filename);
        return;
    }

    // Write PGM header
    fprintf(file, "P2\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "255\n");

    // Normalize and write pixel data
    float max_value = 0.0f;
    for (int i = 0; i < height * width; i++) {
        if (data[i] > max_value) max_value = data[i];
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;
            int value = (int)((data[index] / max_value) * 255); // Normalize to 0-255
            fprintf(file, "%d ", value);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    printf("Saved frame to %s\n", filename);
}

// Generate and save temporal frames for a sample
void visualize_sample_frames(const NMNISTSample *sample, const char *output_dir, int time_bins, int height, int width, unsigned int max_time) {
    // Convert events to discretized input
    float *discretized_input = convert_events_to_input(
        sample->events, sample->num_events, time_bins, height, width, max_time);

    // Create output directory if it doesn't exist
    char mkdir_command[256];
    snprintf(mkdir_command, sizeof(mkdir_command), "mkdir -p %s", output_dir);
    system(mkdir_command);

    // Save each time bin as a separate frame
    for (int t = 0; t < time_bins; t++) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/frame_%03d.pgm", output_dir, t);

        // Extract the 2D frame for the current time bin
        float *frame = &discretized_input[t * height * width];
        save_frame_as_pgm(filename, frame, height, width);
    }

    free(discretized_input);
}

// Free the NMNIST dataset
void free_nmnist_dataset(NMNISTDataset *dataset) {
    for (size_t i = 0; i < dataset->num_samples; i++) {
        free(dataset->samples[i].events);
    }
    free(dataset->samples);
    free(dataset);
}
