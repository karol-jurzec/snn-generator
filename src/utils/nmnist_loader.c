#include "../../include/utils/nmnist_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

typedef struct {
    char path[512];
    int label;
} NMNISTSamplePath;

static void nmnist_to_spike_event(const unsigned char *buffer, SpikeEvent *event) {
    event->x = buffer[0];
    event->y = buffer[1];
    event->polarity = (buffer[2] >> 7) & 1;
    event->timestamp = ((buffer[2] & 0x7F) << 16) | (buffer[3] << 8) | buffer[4];
    event->neuron_id = event->y * NMNIST_WIDTH + event->x;
    event->channel = 0;
}

void normalize_path(char *path) {
#ifdef _WIN32
    for (char *p = path; *p; ++p)
        if (*p == '\\') *p = '/';
#endif
}

Dataset *load_nmnist_dataset(const char *data_dir, size_t max_samples, bool stabilize, bool denoise) {
    Dataset *dataset = create_empty_dataset(max_samples, NMNIST_WIDTH, NMNIST_HEIGHT, NMNIST_CHANNELS, NMNIST_CLASSES, FRAME_BY_N_TIME_BINS, 300);
    if (!dataset) {
        fprintf(stderr, "Error: Failed to create dataset structure\n");
        return NULL;
    }

    size_t base_per_class = max_samples / NMNIST_CLASSES;
    size_t remainder = max_samples % NMNIST_CLASSES;

    for (int digit = 0; digit < NMNIST_CLASSES; digit++) {
        size_t target_samples = base_per_class + (digit < remainder ? 1 : 0);
        if (target_samples == 0) continue;

        char digit_dir[512];
        snprintf(digit_dir, sizeof(digit_dir), "%s/%d", data_dir, digit);

        DIR *dir = opendir(digit_dir);
        if (!dir) {
            fprintf(stderr, "Warning: Could not open directory %s\n", digit_dir);
            continue;
        }

        // First pass: count .bin files
        size_t file_count = 0;
        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {
            if (strstr(entry->d_name, ".bin")) file_count++;
        }
        rewinddir(dir);

        if (file_count == 0) {
            closedir(dir);
            continue;
        }

        NMNISTSamplePath *paths = malloc(file_count * sizeof(NMNISTSamplePath));
        if (!paths) {
            closedir(dir);
            fprintf(stderr, "Error: Memory allocation failed\n");
            free_dataset(dataset);
            return NULL;
        }

        // Collect file paths
        size_t collected = 0;
        while ((entry = readdir(dir)) != NULL && collected < file_count) {
            if (strstr(entry->d_name, ".bin")) {
                snprintf(paths[collected].path, sizeof(paths[collected].path), "%s/%s", digit_dir, entry->d_name);
                paths[collected].label = digit;
                collected++;
            }
        }
        closedir(dir);

        // Shuffle
        for (size_t i = 0; i < collected; i++) {
            size_t j = i + rand() % (collected - i);
            NMNISTSamplePath tmp = paths[i];
            paths[i] = paths[j];
            paths[j] = tmp;
        }

        // Load up to target_samples
        size_t loaded = 0;
        for (size_t i = 0; i < collected && loaded < target_samples && dataset->num_samples < max_samples; i++) {
            FILE *file = fopen(paths[i].path, "rb");
            if (!file) continue;

            fseek(file, 0, SEEK_END);
            size_t file_size = ftell(file);
            fseek(file, 0, SEEK_SET);
            size_t num_events = file_size / 5;

            SpikeEvent *events = malloc(num_events * sizeof(SpikeEvent));
            if (!events) {
                fclose(file);
                free(paths);
                free_dataset(dataset);
                return NULL;
            }

            for (size_t j = 0; j < num_events; j++) {
                unsigned char buffer[5];
                if (fread(buffer, 1, 5, file) != 5) {
                    fclose(file);
                    free(events);
                    continue;
                }
                nmnist_to_spike_event(buffer, &events[j]);
            }
            fclose(file);

            if (stabilize) {
                stabilize_nmnist_events(events, num_events);
            }

            if (denoise) {
                num_events = denoise_events(events, num_events, 10000);
            }

            if (add_sample_to_dataset(dataset, events, num_events, paths[i].label) == 0) {
                loaded++;
            }

            free(events);
        }

        free(paths);
    }

    shuffle_dataset(dataset);

    return dataset;
}
