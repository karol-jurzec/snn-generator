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
    FORMAT_HDF5_SHD,
    FORMAT_HDF5_SSC,
    FORMAT_NCARS,
    FORMAT_DVSGESTURE,
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

// Specific dataset loaders
Dataset *load_nmnist_dataset(const char *dir, size_t max_samples, bool stabilize, bool denoise);

#endif // DATASET_LOADER_H