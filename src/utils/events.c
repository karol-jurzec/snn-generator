#include "../../include/utils/events.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

size_t denoise_events(SpikeEvent *events, size_t num_events, unsigned int filter_time) {
    if (num_events == 0 || filter_time == 0) return num_events;

    SpikeEvent *filtered = (SpikeEvent *)malloc(num_events * sizeof(SpikeEvent));
    if (!filtered) {
        fprintf(stderr, "Error: Memory allocation failed in denoise_events\n");
        return 0;
    }

    size_t filtered_count = 0;

    for (size_t i = 0; i < num_events; i++) {
        bool has_neighbor = false;

        // Check backward neighbor
        if (i > 0 && (events[i].timestamp - events[i-1].timestamp <= filter_time)) {
            has_neighbor = true;
        }

        // Check forward neighbor
        if (i < num_events-1 && (events[i+1].timestamp - events[i].timestamp <= filter_time)) {
            has_neighbor = true;
        }

        if (has_neighbor) {
            filtered[filtered_count++] = events[i];
        }
    }

    // Copy filtered events back
    memcpy(events, filtered, filtered_count * sizeof(SpikeEvent));
    free(filtered);

    return filtered_count;
}

void stabilize_nmnist_events(SpikeEvent *events, size_t num_events) {
    for (size_t i = 0; i < num_events; i++) {
        unsigned int ts = events[i].timestamp;

        if (ts <= 105000) {
            events[i].x -= (3.5f * ts / 105000);
            events[i].y -= (7.0f * ts / 105000);
        } else if (ts <= 210000) {
            events[i].x -= (3.5f + (3.5f * (ts - 105000) / 105000));
            events[i].y -= (7.0f - (7.0f * (ts - 105000) / 105000));
        } else {
            events[i].x -= (7.0f - (7.0f * (ts - 210000) / 105000));
        }

        // Clamp coordinates
        if (events[i].x < 1) events[i].x = 1;
        if (events[i].y < 1) events[i].y = 1;
        
        // Update neuron_id if used
        events[i].neuron_id = events[i].y * 34 + events[i].x;
    }
}

static int16_t ****allocate_frames(int num_bins, int height, int width) {
    int16_t ****frames = malloc(num_bins * sizeof(int16_t ***));
    if (!frames) return NULL;

    for (int b = 0; b < num_bins; b++) {
        frames[b] = malloc(2 * sizeof(int16_t **));
        if (!frames[b]) {
            // Clean up previous allocations
            for (int b_clean = 0; b_clean < b; b_clean++) {
                for (int p = 0; p < 2; p++) {
                    for (int y = 0; y < height; y++) {
                        free(frames[b_clean][p][y]);
                    }
                    free(frames[b_clean][p]);
                }
                free(frames[b_clean]);
            }
            free(frames);
            return NULL;
        }

        for (int p = 0; p < 2; p++) {
            frames[b][p] = malloc(height * sizeof(int16_t *));
            if (!frames[b][p]) {
                // Clean up
                for (int b_clean = 0; b_clean <= b; b_clean++) {
                    for (int p_clean = 0; p_clean < (b_clean == b ? p : 2); p_clean++) {
                        for (int y = 0; y < height; y++) {
                            free(frames[b_clean][p_clean][y]);
                        }
                        free(frames[b_clean][p_clean]);
                    }
                    free(frames[b_clean]);
                }
                free(frames);
                return NULL;
            }

            for (int y = 0; y < height; y++) {
                frames[b][p][y] = calloc(width, sizeof(int16_t));
                if (!frames[b][p][y]) {
                    // Clean up
                    for (int b_clean = 0; b_clean <= b; b_clean++) {
                        for (int p_clean = 0; p_clean < (b_clean == b ? p+1 : 2); p_clean++) {
                            for (int y_clean = 0; y_clean < (b_clean == b && p_clean == p ? y : height); y_clean++) {
                                free(frames[b_clean][p_clean][y_clean]);
                            }
                            free(frames[b_clean][p_clean]);
                        }
                        free(frames[b_clean]);
                    }
                    free(frames);
                    return NULL;
                }
            }
        }
    }
    return frames;
}

int16_t ****events_to_frames(SpikeEvent *events, size_t num_events,
                           int width, int height, FrameSlicingMode mode,
                           int mode_param, float overlap, int *out_num_bins) {
    if (num_events == 0 || mode_param <= 0) return NULL;

    uint32_t t_start = events[0].timestamp;
    uint32_t t_end = events[num_events-1].timestamp;
    uint32_t duration = t_end - t_start;
    if (duration == 0) duration = 1;

    int num_bins = 0;

    switch (mode) {
        case FRAME_BY_N_TIME_BINS:
            num_bins = mode_param;
            break;
            
        case FRAME_BY_TIME_WINDOW: {
            uint32_t window = (uint32_t)mode_param;
            uint32_t stride = (uint32_t)(window * (1.0f - overlap));
            if (stride == 0) stride = 1;

            for (uint32_t t = t_start; t + window <= t_end; t += stride)
                num_bins++;
            break;
        }
            
        case FRAME_BY_EVENT_COUNT: {
            size_t stride = (size_t)(mode_param * (1.0f - overlap));
            if (stride == 0) stride = 1;
            num_bins = (num_events - mode_param) / stride + 1;
            break;
        }
            
        default:
            return NULL;
    }

    if (num_bins <= 0) return NULL;
    if (out_num_bins) *out_num_bins = num_bins;

    int16_t ****frames = allocate_frames(num_bins, height, width);
    if (!frames) return NULL;

    // Process events per mode
    switch (mode) {
        case FRAME_BY_N_TIME_BINS: {
            for (size_t i = 0; i < num_events; i++) {
                SpikeEvent e = events[i];
                int bin = (int)(((e.timestamp - t_start) * num_bins) / (float)duration);
                if (bin >= num_bins) bin = num_bins-1;
                if (e.polarity > 1 || e.x >= width || e.y >= height) continue;
                frames[bin][e.polarity][e.y][e.x] += 1;
            }
            break;
        }

        case FRAME_BY_TIME_WINDOW: {
            uint32_t window = (uint32_t)mode_param;
            uint32_t stride = (uint32_t)(window * (1.0f - overlap));
            if (stride == 0) stride = 1;

            size_t idx = 0;
            for (int bin = 0; bin < num_bins; bin++) {
                uint32_t bin_start = t_start + bin * stride;
                uint32_t bin_end = bin_start + window;

                while (idx < num_events && events[idx].timestamp < bin_start) idx++;

                for (size_t j = idx; j < num_events && events[j].timestamp < bin_end; j++) {
                    SpikeEvent e = events[j];
                    if (e.polarity > 1 || e.x >= width || e.y >= height) continue;
                    int val = e.polarity;
                    frames[bin][e.polarity][e.y][e.x] += 1;
                }
            }
            break;
        }

        case FRAME_BY_EVENT_COUNT: {
            size_t stride = (size_t)(mode_param * (1.0f - overlap));
            if (stride == 0) stride = 1;

            for (int bin = 0; bin < num_bins; bin++) {
                size_t start = bin * stride;
                size_t end = start + mode_param;
                if (end > num_events) end = num_events;

                for (size_t j = start; j < end; j++) {
                    SpikeEvent e = events[j];
                    if (e.polarity > 1 || e.x >= width || e.y >= height) continue;
                    frames[bin][e.polarity][e.y][e.x] += 1;
                }
            }
            break;
        }
    }

    return frames;
}

void free_frames(int16_t ****frames, int num_bins, int height) {
    if (!frames) return;
    
    for (int b = 0; b < num_bins; b++) {
        if (!frames[b]) continue;
        
        for (int p = 0; p < 2; p++) {
            if (!frames[b][p]) continue;
            
            for (int y = 0; y < height; y++) {
                free(frames[b][p][y]);
            }
            free(frames[b][p]);
        }
        free(frames[b]);
    }
    free(frames);
}

void zero_time_bin(int16_t ****frames, int bin_to_zero, int width, int height) {
    for (int c = 0; c < 2; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                frames[bin_to_zero][c][y][x] = 0;
            }
        }
    }
}

float *flatten_frames_to_float(int16_t ****frames, int bins, int height, int width) {
    if (!frames || bins <= 0 || height <= 0 || width <= 0) return NULL;

    size_t size = (size_t)bins * 2 * height * width;
    float *output = (float *)calloc(size, sizeof(float));
    if (!output) {
        fprintf(stderr, "Error: Memory allocation failed in flatten_frames_to_float\n");
        return NULL;
    }

    //zero_time_bin(frames, 25, height, width);

    for (int b = 0; b < bins; b++) {
        for (int c = 0; c < 2; c++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    // size_t idx = ((size_t)c * bins * height * width) +
                    //              ((size_t)b * height * width) +
                    //              ((size_t)y * width) + x;

                    size_t idx = ((size_t)b * 2 * height * width) +
                                 ((size_t)c * height * width) +
                                 ((size_t)y * width) + x;
                    


                    output[idx] = (float)frames[b][c][y][x];
                }
            }
        }
    }

    return output;
}