#ifndef EVENTS_H
#define EVENTS_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

// Generic spike event structure
typedef struct {
    unsigned int timestamp;    // Time of the spike in microseconds
    unsigned short x;          // X-coordinate (if applicable)
    unsigned short y;          // Y-coordinate (if applicable)
    unsigned short neuron_id;  // Encoded position (alternative to x,y)
    unsigned char polarity;    // Event polarity (0/1 or other values)
    unsigned char channel;     // Optional channel identifier
} SpikeEvent;

// Frame slicing modes
typedef enum {
    FRAME_BY_TIME_WINDOW,
    FRAME_BY_EVENT_COUNT,
    FRAME_BY_N_TIME_BINS
} FrameSlicingMode;

// Event processing functions
size_t denoise_events(SpikeEvent *events, size_t num_events, unsigned int filter_time);
void stabilize_nmnist_events(SpikeEvent *events, size_t num_events);

// Frame conversion functions
int16_t ****events_to_frames(SpikeEvent *events, size_t num_events,
                           int width, int height, FrameSlicingMode mode,
                           int mode_param, float overlap, int *out_num_bins);

void free_frames(int16_t ****frames, int num_bins, int height);
float *flatten_frames_to_float(int16_t ****frames, int bins, int height, int width);

#endif // EVENTS_H