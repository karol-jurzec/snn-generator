#include "../../include/utils/stmnist_loader.h"
#include <sys/stat.h>


// Forward declarations
static int scan_directory_for_mat_files(const char *base_dir, char ***paths, int **labels, size_t *count);
static int load_stmnist_sample(const char *file_path, int label, Dataset *dataset);

Dataset *load_stmnist_dataset(const char *dir, size_t max_samples, bool stabilize, bool denoise) {
    char **file_paths = NULL;
    int *labels = NULL;
    size_t total_files = 0;

    if (scan_directory_for_mat_files(dir, &file_paths, &labels, &total_files) != 0) {
        fprintf(stderr, "Failed to scan STMNIST directory\n");
        return NULL;
    }

    // Organize file paths by class
    const int num_classes = STMNIST_CLASSES;
    size_t per_class_max = (max_samples > 0) ? max_samples / num_classes : SIZE_MAX;

    size_t class_counts[STMNIST_CLASSES] = {0};
    size_t class_capacities[STMNIST_CLASSES] = {0};
    char ***class_files = calloc(num_classes, sizeof(char **));

    for (int i = 0; i < num_classes; ++i) {
        class_capacities[i] = 128;
        class_files[i] = malloc(class_capacities[i] * sizeof(char *));
    }

    for (size_t i = 0; i < total_files; ++i) {
        int label = labels[i];
        if (label < 0 || label >= num_classes) continue;

        if (class_counts[label] >= class_capacities[label]) {
            class_capacities[label] *= 2;
            class_files[label] = realloc(class_files[label], class_capacities[label] * sizeof(char *));
        }
        class_files[label][class_counts[label]++] = file_paths[i];
    }

    // Estimate actual sample count
    size_t samples_to_load = 0;
    for (int i = 0; i < num_classes; ++i) {
        if (class_counts[i] < per_class_max) per_class_max = class_counts[i]; 
    }
    samples_to_load = per_class_max * num_classes;

    Dataset *dataset = create_empty_dataset(samples_to_load, STMNIST_WIDTH, STMNIST_HEIGHT, STMNIST_CHANNELS, STMNIST_CLASSES, FRAME_BY_TIME_WINDOW, 20000);
    if (!dataset) return NULL;

    for (int i = 0; i < num_classes; ++i) {
        // Shuffle each class's file list
        for (size_t j = 0; j < class_counts[i]; ++j) {
            size_t rand_idx = j + rand() % (class_counts[i] - j);
            char *tmp = class_files[i][j];
            class_files[i][j] = class_files[i][rand_idx];
            class_files[i][rand_idx] = tmp;
        }

        // Load up to per_class_max samples for this class
        for (size_t j = 0; j < per_class_max; ++j) {
            const char *path = class_files[i][j];
            if (strstr(path, "LUT")) continue;

            if (load_stmnist_sample(path, i, dataset) != 0) {
                fprintf(stderr, "Failed to load sample: %s\n", path);
            }
            free((void *)path); // Clean up memory from scan_directory
        }
        free(class_files[i]);
    }

    free(class_files);
    free(file_paths);
    free(labels);

    shuffle_dataset(dataset);
    return dataset;
}

static int scan_directory_for_mat_files(const char *base_dir, char ***paths, int **labels, size_t *count) {
    size_t capacity = 512;
    *paths = malloc(capacity * sizeof(char *));
    *labels = malloc(capacity * sizeof(int));
    *count = 0;

    if (!*paths || !*labels) return -1;

    DIR *root = opendir(base_dir);
    if (!root) return -1;

    struct dirent *entry;
    while ((entry = readdir(root))) {
        char class_dir[1024];
        snprintf(class_dir, sizeof(class_dir), "%s/%s", base_dir, entry->d_name);

        struct stat path_stat;
        if (stat(class_dir, &path_stat) != 0 || !S_ISDIR(path_stat.st_mode)) continue;
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;

        snprintf(class_dir, sizeof(class_dir), "%s/%s", base_dir, entry->d_name);
        int label = atoi(entry->d_name);

        DIR *sub = opendir(class_dir);
        if (!sub) continue;

        struct dirent *fentry;
        while ((fentry = readdir(sub))) {
            if (strstr(fentry->d_name, ".mat") == NULL) continue;

            char *full_path = malloc(1024);
            snprintf(full_path, 1024, "%s/%s", class_dir, fentry->d_name);

            if (*count >= capacity) {
                capacity *= 2;
                *paths = realloc(*paths, capacity * sizeof(char *));
                *labels = realloc(*labels, capacity * sizeof(int));
            }

            (*paths)[*count] = full_path;
            (*labels)[*count] = label;
            (*count)++;
        }
        closedir(sub);
    }
    closedir(root);
    return 0;
}

static int load_stmnist_sample(const char *file_path, int label, Dataset *dataset) {
    mat_t *matfp = Mat_Open(file_path, MAT_ACC_RDONLY);
    if (!matfp) {
        fprintf(stderr, "Failed to open MAT file: %s\n", file_path);
        return -1;
    }

    matvar_t *var = Mat_VarRead(matfp, "spiketrain");
    if (!var || var->rank != 2 || var->data_type != MAT_T_DOUBLE) {
        fprintf(stderr, "Invalid spiketrain variable in: %s\n", file_path);
        if (var) Mat_VarFree(var);
        Mat_Close(matfp);
        return -1;
    }

    // Dimensions: [101 x num_events]
    size_t num_rows = var->dims[0];    // Should be 101 (100 taxels + 1 timestamp row)
    size_t num_events = var->dims[1];  // Number of events in this sample
    
    if (num_rows != 101) {
        fprintf(stderr, "Unexpected matrix shape in %s: %zux%zu (expected 101xn)\n", 
                file_path, num_rows, num_events);
        Mat_VarFree(var);
        Mat_Close(matfp);
        return -1;
    }

    double *spiketrain = (double *)var->data;

    // First pass to count actual events (non-zero in taxel rows)
    size_t valid_events = 0;
    for (size_t e = 0; e < num_events; e++) {
        for (size_t t = 0; t < 100; t++) { // Check only taxel rows
            if (spiketrain[t + e * num_rows] != 0.0) {
                valid_events++;
            }
        }
    }

    if (valid_events == 0) {
        fprintf(stderr, "No valid events found in: %s\n", file_path);
        Mat_VarFree(var);
        Mat_Close(matfp);
        return -1;
    }

    // Allocate events
    SpikeEvent *events = malloc(valid_events * sizeof(SpikeEvent));
    if (!events) {
        Mat_VarFree(var);
        Mat_Close(matfp);
        return -1;
    }

    // Extract events
    size_t event_idx = 0;
    for (size_t e = 0; e < num_events; e++) {
        double timestamp = spiketrain[100 + e * num_rows]; // 101st row is timestamp
        
        for (size_t t = 0; t < 100; t++) {
            double val = spiketrain[t + e * num_rows];
            if (val == 0.0) continue;

            int x = t % 10;
            int y = t / 10;
            int p = val > 0 ? 1 : 0; // 1 for ON, 0 for OFF

            SpikeEvent evt = {
                .x = x,
                .y = y,
                .timestamp = (uint64_t)(timestamp * 1e6), // Convert to microseconds
                .polarity = (int8_t)p
            };
            events[event_idx++] = evt;
        }
    }

    int result = add_sample_to_dataset(dataset, events, valid_events, label);
    free(events);
    Mat_VarFree(var);
    Mat_Close(matfp);
    return result;
}