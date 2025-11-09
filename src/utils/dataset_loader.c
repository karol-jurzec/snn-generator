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