#ifndef NMNIST_LOADER_H
#define NMNIST_LOADER_H

#include <stddef.h>
#include <stdbool.h>

// NMNIST spike event structure
typedef struct {
    unsigned int timestamp;  // Time of the spike in microseconds
    unsigned char x;         // X-coordinate of the spike
    unsigned char y;         // Y-coordinate of the spike
    unsigned char polarity;  // Polarity (0 = OFF, 1 = ON)
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
NMNISTDataset *load_nmnist_dataset(const char *data_dir, size_t max_samples, bool stabilize, int num_classes);
NMNISTSample load_nmnist_sample(const char *file_path, int label, bool stabilize);
void free_nmnist_dataset(NMNISTDataset *dataset);
float *convert_events_to_input(const NMNISTEvent *events, size_t num_events, int time_bins, int height, int width, unsigned int max_time);

void save_frame_as_pgm(const char *filename, float *data, int height, int width);
void save_frame_as_image(const char *filename, float *data, int height, int width, int is_png);
void visualize_sample_frames(const NMNISTSample *sample, const char *output_dir, int time_bins, int height, int width, unsigned int max_time);
void print_frame(float* data, int width, int height);
void plot_event_grid(NMNISTSample* events, int axis_x, int axis_y, int plot_frame_number);
unsigned char max_value(NMNISTEvent* events, int num_events, char field);





#endif // NMNIST_LOADER_H
