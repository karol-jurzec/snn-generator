#ifndef DATASET_LOADER_H
#define DATASET_LOADER_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

// Generalized SpikeEvent structure (common for all datasets)
typedef struct {
    unsigned int timestamp;    // Time of the spike (microseconds)
    unsigned short neuron_id;  // Generalized from (x, y) or other encoding
    unsigned char polarity;    // Polarity (optional)
} SpikeEvent;

// Generalized Sample structure (works for any dataset)
typedef struct {
    size_t num_events;
    SpikeEvent *events;
    int label;
    int16_t ****frames;  // 4D frame representation (time x polarity x height x width)
    float *input;        // Flattened input for machine learning models
    int num_bins;        // Number of time bins
} Sample;

// Generalized Dataset structure
typedef struct {
    Sample *samples;
    size_t num_samples;
} Dataset;

typedef enum {
    FORMAT_NMNIST,
    FORMAT_HDF5_SHD,
    FORMAT_HDF5_SSC,
    FORMAT_NCARS,
    FORMAT_DVSGESTURE,
    FORMAT_UNKNOWN
} DatasetFormat;

// Dataset loading interface
Dataset *load_dataset(const char *source_path, DatasetFormat format, 
                     size_t max_samples, bool stabilize, bool denoise);

// Specific dataset loaders
Dataset *load_nmnist_dataset(const char *dir, size_t max_samples, bool stabilize, bool denoise);

#endif // DATASET_LOADER_H