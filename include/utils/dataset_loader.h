#ifndef DATASET_LOADER_H
#define DATASET_LOADER_H

#include "events.h"
#include <stddef.h>

typedef struct {
    size_t num_events;
    SpikeEvent *events;
    int label;
    int16_t ****frames;    // opt frame representation
    float *input;         // flattened input for neural networks
    int num_bins;        // no of time bins if frames are used
} Sample;

typedef struct {
    Sample *samples;
    size_t num_samples;
    FrameSlicingMode slicing_mode; // mode in toFrame method
    int mode_param;           // param_mode
    int input_channels;
    int input_width;          // sensor width
    int input_height;         // sensor height
    int num_classes;          // no of classes
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

// main methods
Dataset *create_empty_dataset(size_t initial_capacity, int width, int height, int channels, int num_classes, FrameSlicingMode mode, int mode_param);
void free_dataset(Dataset *dataset);
int add_sample_to_dataset(Dataset *dataset, const SpikeEvent *events, size_t num_events, int label);

// dataset loading interface
Dataset *load_dataset(const char *source_path, DatasetFormat format, 
                     size_t max_samples, bool stabilize, bool denoise);

// specific dataset loaders
Dataset *load_nmnist_dataset(const char *dir, size_t max_samples, bool stabilize, bool denoise);

#endif // DATASET_LOADER_H