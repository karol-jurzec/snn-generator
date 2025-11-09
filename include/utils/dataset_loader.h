#ifndef DATASET_LOADER_H
#define DATASET_LOADER_H

#include "events.h"
#include <stddef.h>

typedef struct {
    size_t num_events;
    SpikeEvent *events;
    int label;
    int16_t ****frames;    // Optional frame representation
    float *input;         // Flattened input for neural networks
    int num_bins;        // Number of time bins if frames are used
} Sample;

typedef struct {
    Sample *samples;
    size_t num_samples;
    FrameSlicingMode slicing_mode; // mode in toFrame method
    int mode_param;        // param_mode
    int input_channels;
    int input_width;          // Sensor width
    int input_height;         // Sensor height
    int num_classes;    // Number of classes
} Dataset;

typedef enum {
    FORMAT_NMNIST,
    FORMAT_STMNIST,
    FORMAT_UNKNOWN
} DatasetFormat;

// Core dataset functions
Dataset *create_empty_dataset(size_t initial_capacity, int width, int height, int channels, int num_classes, FrameSlicingMode mode, int mode_param);
void free_dataset(Dataset *dataset);
int add_sample_to_dataset(Dataset *dataset, const SpikeEvent *events, size_t num_events, int label);

// Dataset loading interface
Dataset *load_dataset(const char *source_path, DatasetFormat format, 
                     size_t max_samples, bool stabilize, bool denoise);

                     // Load single sample from file
Dataset *load_single_sample_file(const char *file_path, int label, DatasetFormat format, bool stabilize, bool denoise);

// Detect format from file extension
DatasetFormat detect_format_from_file(const char *file_path);

// Specific dataset loaders
#endif // DATASET_LOADER_H