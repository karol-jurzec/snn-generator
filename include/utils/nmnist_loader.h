#ifndef NMNIST_LOADER_H
#define NMNIST_LOADER_H

#include <stddef.h>

// NMNIST spike event structure
typedef struct {
    unsigned int timestamp;  // Time of the spike (in ms or some time unit)
    unsigned char x;         // X-coordinate of the spike
    unsigned char y;         // Y-coordinate of the spike
    unsigned char polarity;  // Polarity (0 = off, 1 = on)
} NMNISTEvent;

// NMNIST sample structure (spike trains for one digit)
typedef struct {
    size_t num_events;       // Number of spike events
    NMNISTEvent *events;     // Array of spike events
    int label;               // Class label (0-9)
} NMNISTSample;

// NMNIST dataset structure
typedef struct {
    NMNISTSample *samples;   // Array of NMNIST samples
    size_t num_samples;      // Number of samples in the dataset
} NMNISTDataset;

// Function declarations
NMNISTDataset *load_nmnist_dataset(const char *data_dir, size_t max_samples);
void free_nmnist_dataset(NMNISTDataset *dataset);

#endif // NMNIST_LOADER_H
