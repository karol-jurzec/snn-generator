#include "../../include/utils/dataset_loader.h"
#include "../../include/utils/events.h"
#include "../../include/utils/stmnist_loader.h"
#include "../../include/utils/nmnist_loader.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

Dataset *create_empty_dataset(size_t initial_capacity, int width, int height, int channels, int num_classes, FrameSlicingMode mode, int mode_param) {
    Dataset *dataset = malloc(sizeof(Dataset));
    if (!dataset) return NULL;

    dataset->samples = calloc(initial_capacity, sizeof(Sample));
    if (!dataset->samples) {
        free(dataset);
        return NULL;
    }

    dataset->num_samples = 0;
    dataset->input_channels = channels;
    dataset->input_width = width;
    dataset->input_height = height;
    dataset->num_classes = num_classes;
    dataset->slicing_mode = mode;
    dataset->mode_param = mode_param;
    
    
    return dataset;
}

void free_dataset(Dataset *dataset) {
    if (!dataset) return;

    for (size_t i = 0; i < dataset->num_samples; i++) {
        free(dataset->samples[i].events);
        free_frames(dataset->samples[i].frames, dataset->samples[i].num_bins, dataset->input_height);
        free(dataset->samples[i].input);
    }
    
    free(dataset->samples);
    free(dataset);
}

int add_sample_to_dataset(Dataset *dataset, const SpikeEvent *events, size_t num_events, int label) {
    if (!dataset || !events || num_events == 0) return -1;

    // reallocate if needed 
    if (dataset->num_samples >= (size_t)-1 / sizeof(Sample)) return -1;
    
    Sample *new_samples = realloc(dataset->samples, (dataset->num_samples + 1) * sizeof(Sample));
    if (!new_samples) return -1;
    
    dataset->samples = new_samples;
    size_t idx = dataset->num_samples++;

    // copy events
    dataset->samples[idx].events = malloc(num_events * sizeof(SpikeEvent));
    if (!dataset->samples[idx].events) {
        dataset->num_samples--;
        return -1;
    }
    memcpy(dataset->samples[idx].events, events, num_events * sizeof(SpikeEvent));

    dataset->samples[idx].num_events = num_events;
    dataset->samples[idx].label = label;
    dataset->samples[idx].frames = events_to_frames(events, num_events,
         dataset->input_width, dataset->input_height, dataset->slicing_mode, dataset->mode_param, 0.0f, &dataset->samples[idx].num_bins);
    dataset->samples[idx].input = flatten_frames_to_float(dataset->samples[idx].frames, dataset->samples[idx].num_bins, dataset->input_height, dataset->input_width);

    return 0;
}

Dataset *load_dataset(const char *source_path, DatasetFormat format, size_t max_samples, bool stabilize, bool denoise) {
    switch (format) {
        case FORMAT_NMNIST:
            return load_nmnist_dataset(source_path, max_samples, stabilize, denoise);
        case FORMAT_STMNIST:
            return load_stmnist_dataset(source_path, max_samples, stabilize, denoise);
        default:
            fprintf(stderr, "Unsupported dataset format\n");
            return NULL;
    }
}

DatasetFormat detect_format_from_file(const char *file_path) {
    if (!file_path) return FORMAT_UNKNOWN;
    
    const char *ext = strrchr(file_path, '.');
    if (!ext) return FORMAT_UNKNOWN;
    
    if (strcmp(ext, ".bin") == 0) {
        return FORMAT_NMNIST;
    } else if (strcmp(ext, ".mat") == 0) {
        return FORMAT_STMNIST;
    }
    
    return FORMAT_UNKNOWN;
}

Dataset *load_single_sample_file(const char *file_path, int label, DatasetFormat format, bool stabilize, bool denoise) {
    if (!file_path) {
        fprintf(stderr, "Error: file_path is NULL\n");
        return NULL;
    }

    if (format == FORMAT_UNKNOWN) {
        format = detect_format_from_file(file_path);
        if (format == FORMAT_UNKNOWN) {
            fprintf(stderr, "Error: Could not detect format from file: %s\n", file_path);
            return NULL;
        }
    }

    Dataset *dataset = NULL;
    
    switch (format) {
        case FORMAT_NMNIST: {
            dataset = create_empty_dataset(1, NMNIST_WIDTH, NMNIST_HEIGHT, NMNIST_CHANNELS, NMNIST_CLASSES, FRAME_BY_N_TIME_BINS, 50);
            if (!dataset) return NULL;

            FILE *file = fopen(file_path, "rb");
            if (!file) {
                fprintf(stderr, "Error: Failed to open file: %s\n", file_path);
                free_dataset(dataset);
                return NULL;
            }

            fseek(file, 0, SEEK_END);
            size_t file_size = ftell(file);
            fseek(file, 0, SEEK_SET);
            size_t num_events = file_size / 5;

            SpikeEvent *events = malloc(num_events * sizeof(SpikeEvent));
            if (!events) {
                fclose(file);
                free_dataset(dataset);
                return NULL;
            }

            unsigned char buffer[5];
            for (size_t j = 0; j < num_events; j++) {
                if (fread(buffer, 1, 5, file) != 5) {
                    fclose(file);
                    free(events);
                    free_dataset(dataset);
                    return NULL;
                }
                
                events[j].x = buffer[0];
                events[j].y = buffer[1];
                events[j].polarity = (buffer[2] >> 7) & 1;
                events[j].timestamp = ((buffer[2] & 0x7F) << 16) | (buffer[3] << 8) | buffer[4];
                events[j].neuron_id = events[j].y * NMNIST_WIDTH + events[j].x;
                events[j].channel = 0;
            }
            fclose(file);

            if (stabilize) {
                stabilize_nmnist_events(events, num_events);
            }

            if (denoise) {
                num_events = denoise_events(events, num_events, 10000);
            }

            if (add_sample_to_dataset(dataset, events, num_events, label) != 0) {
                free(events);
                free_dataset(dataset);
                return NULL;
            }

            free(events);
            break;
        }
        
        case FORMAT_STMNIST: {
            dataset = create_empty_dataset(1, STMNIST_WIDTH, STMNIST_HEIGHT, STMNIST_CHANNELS, STMNIST_CLASSES, FRAME_BY_TIME_WINDOW, 10000);
            if (!dataset) return NULL;

            mat_t *matfp = Mat_Open(file_path, MAT_ACC_RDONLY);
            if (!matfp) {
                fprintf(stderr, "Failed to open MAT file: %s\n", file_path);
                free_dataset(dataset);
                return NULL;
            }

            matvar_t *var = Mat_VarRead(matfp, "spiketrain");
            if (!var || var->rank != 2 || var->data_type != MAT_T_DOUBLE) {
                fprintf(stderr, "Invalid spiketrain variable in: %s\n", file_path);
                if (var) Mat_VarFree(var);
                Mat_Close(matfp);
                free_dataset(dataset);
                return NULL;
            }

            size_t num_rows = var->dims[0];    // Should be 101
            size_t num_events = var->dims[1];
            
            if (num_rows != 101) {
                fprintf(stderr, "Unexpected matrix shape in %s: %zux%zu (expected 101xn)\n", 
                        file_path, num_rows, num_events);
                Mat_VarFree(var);
                Mat_Close(matfp);
                free_dataset(dataset);
                return NULL;
            }

            double *spiketrain = (double *)var->data;

            // Count valid events
            size_t valid_events = 0;
            for (size_t e = 0; e < num_events; e++) {
                for (size_t t = 0; t < 100; t++) {
                    if (spiketrain[t + e * num_rows] != 0.0) {
                        valid_events++;
                    }
                }
            }

            if (valid_events == 0) {
                fprintf(stderr, "No valid events found in: %s\n", file_path);
                Mat_VarFree(var);
                Mat_Close(matfp);
                free_dataset(dataset);
                return NULL;
            }

            // Allocate and extract events
            SpikeEvent *events = malloc(valid_events * sizeof(SpikeEvent));
            if (!events) {
                Mat_VarFree(var);
                Mat_Close(matfp);
                free_dataset(dataset);
                return NULL;
            }

            size_t event_idx = 0;
            for (size_t e = 0; e < num_events; e++) {
                double timestamp = spiketrain[100 + e * num_rows];
                
                for (size_t t = 0; t < 100; t++) {
                    double val = spiketrain[t + e * num_rows];
                    if (val == 0.0) continue;

                    int x = t % 10;
                    int y = t / 10;
                    int p = val > 0 ? 1 : 0;

                    events[event_idx].x = x;
                    events[event_idx].y = y;
                    events[event_idx].timestamp = (uint64_t)(timestamp * 1e6);
                    events[event_idx].polarity = (int8_t)p;
                    events[event_idx].neuron_id = y * STMNIST_WIDTH + x;
                    events[event_idx].channel = 0;
                    event_idx++;
                }
            }

            if (add_sample_to_dataset(dataset, events, valid_events, label) != 0) {
                free(events);
                Mat_VarFree(var);
                Mat_Close(matfp);
                free_dataset(dataset);
                return NULL;
            }

            free(events);
            Mat_VarFree(var);
            Mat_Close(matfp);
            break;
        }
        
        default:
            fprintf(stderr, "Unsupported format for single file loading\n");
            return NULL;
    }

    return dataset;
}