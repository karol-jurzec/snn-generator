#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <dirent.h>
#include <stdint.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define FILTER_TIME 10000  // 10 ms


#include "../../include/utils/stb_image_write.h"
#include "../../include/utils/nmnist_loader.h"

// Load the entire NMNIST dataset
typedef struct {
    char path[512];
    int label;
} SamplePath;

static void stabilize_events(NMNISTEvent *events, size_t num_events) {
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

        // Clamp coordinates and polarity
        if (events[i].x < 1) events[i].x = 1;
        if (events[i].y < 1) events[i].y = 1;
    }
}

size_t denoise_events(NMNISTEvent *events, size_t num_events) {
    NMNISTEvent *filtered = (NMNISTEvent *)malloc(num_events * sizeof(NMNISTEvent));
    if (!filtered) {
        fprintf(stderr, "Error: Memory allocation failed in denoise_events\n");
        return 0;
    }

    size_t filtered_count = 0;

    for (size_t i = 0; i < num_events; i++) {
        bool has_neighbor = false;

        // Check backward neighbor
        if (i > 0 && (events[i].timestamp - events[i - 1].timestamp <= FILTER_TIME)) {
            has_neighbor = true;
        }

        // Check forward neighbor
        if (i < num_events - 1 && (events[i + 1].timestamp - events[i].timestamp <= FILTER_TIME)) {
            has_neighbor = true;
        }

        if (has_neighbor) {
            filtered[filtered_count++] = events[i];
        }
    }

    // Copy filtered events back
    memcpy(events, filtered, filtered_count * sizeof(NMNISTEvent));
    free(filtered);

    return filtered_count;
}

typedef enum {
    FRAME_BY_TIME_WINDOW,
    FRAME_BY_EVENT_COUNT,
    FRAME_BY_N_TIME_BINS
} FrameSlicingMode;

// Frame allocation helper
int16_t ****allocate_frames(int num_bins, int height, int width) {
    int16_t ****frames = malloc(num_bins * sizeof(int16_t ***));
    for (int b = 0; b < num_bins; b++) {
        frames[b] = malloc(2 * sizeof(int16_t **));
        for (int p = 0; p < 2; p++) {
            frames[b][p] = malloc(height * sizeof(int16_t *));
            for (int y = 0; y < height; y++) {
                frames[b][p][y] = calloc(width, sizeof(int16_t));
            }
        }
    }
    return frames;
}

// Frame deallocation helper
void free_frames(int16_t ****frames, int num_bins, int height) {
    for (int b = 0; b < num_bins; b++) {
        for (int p = 0; p < 2; p++) {
            for (int y = 0; y < height; y++) {
                free(frames[b][p][y]);
            }
            free(frames[b][p]);
        }
        free(frames[b]);
    }
    free(frames);
}

int16_t ****to_frame(
    NMNISTEvent *events,
    size_t num_events,
    int width,
    int height,
    int mode,
    int mode_param,
    float overlap,
    int *out_num_bins 
) {
    if (num_events == 0 || mode_param <= 0) return NULL;

    uint32_t t_start = events[0].timestamp;
    uint32_t t_end = events[num_events - 1].timestamp;
    uint32_t duration = t_end - t_start;
    if (duration == 0) duration = 1;

    int num_bins = 0;
    size_t *event_start_indices = NULL;

    switch (mode) {
        case FRAME_BY_N_TIME_BINS: {
            num_bins = mode_param;
            break;
        }
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

    // Process events per mode
    switch (mode) {
        case FRAME_BY_N_TIME_BINS: {
            for (size_t i = 0; i < num_events; i++) {
                NMNISTEvent e = events[i];
                int bin = (int)(((e.timestamp - t_start) * num_bins) / (float)duration);
                if (bin >= num_bins) bin = num_bins - 1;
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
                    NMNISTEvent e = events[j];
                    if (e.polarity > 1 || e.x >= width || e.y >= height) continue;
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
                    NMNISTEvent e = events[j];
                    if (e.polarity > 1 || e.x >= width || e.y >= height) continue;
                    frames[bin][e.polarity][e.y][e.x] += 1;
                }
            }
            break;
        }
    }

    return frames;
}

float *flatten_frames_to_float(int16_t ****frames, int bins, int height, int width) {
    int channels = 2;
    size_t size = (size_t)bins * channels * height * width;
    float *output = (float *)calloc(size, sizeof(float));
    if (!output) {
        fprintf(stderr, "Error: Memory allocation failed in flatten_frames_to_float\n");
        return NULL;
    }

    for (int b = 0; b < bins; b++) {
        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    size_t idx = ((size_t)c * bins * height * width) +
                                 ((size_t)b * height * width) +
                                 ((size_t)y * width) + x;
                    output[idx] = (float)frames[b][c][y][x];
                }
            }
        }
    }

    return output;
}

NMNISTSample load_nmnist_sample(const char *file_path, int label, bool stabilize, bool denoise) {
    FILE *file = fopen(file_path, "rb");
    if (!file) {
        printf("Error: Could not open file %s\n", file_path);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    size_t num_events = file_size / 5;
    NMNISTEvent *events = (NMNISTEvent *)malloc(num_events * sizeof(NMNISTEvent));
    if (!events) {
        printf("Error: Memory allocation for NMNIST events failed\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < num_events; i++) {
        unsigned char buffer[5];
        fread(buffer, 1, 5, file);

        events[i].x = buffer[0];
        events[i].y = buffer[1];
        events[i].polarity = (buffer[2] >> 7) & 1;
        events[i].timestamp = ((buffer[2] & 0x7F) << 16) | (buffer[3] << 8) | buffer[4];
    }

    fclose(file);

    if (stabilize) {
        stabilize_events(events, num_events);
    }

    if (denoise) {
        num_events = denoise_events(events, num_events);
    }

    NMNISTSample sample;
    sample.num_events = num_events;
    sample.events = events;
    sample.label = label;

    // Generate 4D frame representation
    int num_bins;
    sample.frames = to_frame(
        events,
        num_events,
        34, 34,                      // Width x Height for NMNIST
        FRAME_BY_N_TIME_BINS,        // Or whatever slicing mode you prefer
        300,                           // e.g. 10 bins
        0.0f,                        // No overlap
        &num_bins
    );
    sample.num_bins = num_bins;

    // Flatten to float* input for the neural network
    sample.input = flatten_frames_to_float(sample.frames, num_bins, 34, 34);

    return sample;
}

NMNISTDataset *load_nmnist_dataset(const char *data_dir, size_t max_samples, bool stabilize, bool denoise, int num_classes) {
    if (num_classes < 1 || num_classes > 10) {
        printf("Error: num_classes must be between 1 and 10\n");
        return NULL;
    }

    size_t max_per_class = max_samples / num_classes;
    if (max_per_class == 0) {
        printf("Error: max_samples is too small for the number of classes\n");
        return NULL;
    }

    NMNISTDataset *dataset = (NMNISTDataset *)malloc(sizeof(NMNISTDataset));
    if (!dataset) {
        printf("Error: Memory allocation for NMNIST dataset failed\n");
        exit(EXIT_FAILURE);
    }

    dataset->samples = (NMNISTSample *)malloc(max_samples * sizeof(NMNISTSample));
    if (!dataset->samples) {
        printf("Error: Memory allocation for NMNIST samples failed\n");
        free(dataset);
        exit(EXIT_FAILURE);
    }

    dataset->num_samples = 0;

    for (int digit = 0; digit < num_classes; digit++) {
        char digit_dir[256];
        snprintf(digit_dir, sizeof(digit_dir), "%s/%d", data_dir, digit);

        DIR *dir = opendir(digit_dir);
        if (!dir) {
            printf("Warning: Could not open directory %s\n", digit_dir);
            continue;
        }

        // Collect all .bin files for this class
        SamplePath *class_paths = NULL;
        size_t class_count = 0;

        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {
            if (strstr(entry->d_name, ".bin")) {
                class_paths = realloc(class_paths, (class_count + 1) * sizeof(SamplePath));
                snprintf(class_paths[class_count].path, sizeof(class_paths[class_count].path),
                         "%s/%s", digit_dir, entry->d_name);
                class_paths[class_count].label = digit;
                class_count++;
            }
        }
        closedir(dir);

        // Shuffle this class's paths
        for (size_t i = 0; i < class_count; i++) {
            size_t j = rand() % class_count;
            SamplePath tmp = class_paths[i];
            class_paths[i] = class_paths[j];
            class_paths[j] = tmp;
        }

        // Load up to max_per_class from this class
        size_t samples_loaded = 0;
        for (size_t i = 0; i < class_count && samples_loaded < max_per_class; i++) {
            NMNISTSample sample = load_nmnist_sample(class_paths[i].path, class_paths[i].label, stabilize, denoise);
            dataset->samples[dataset->num_samples++] = sample;
            samples_loaded++;
        }

        free(class_paths);
    }

        // Shuffle the entire dataset
    for (size_t i = 0; i < dataset->num_samples; i++) {
        size_t j = rand() % dataset->num_samples;
        NMNISTSample tmp = dataset->samples[i];
        dataset->samples[i] = dataset->samples[j];
        dataset->samples[j] = tmp;
    }

    return dataset;
}

float *load_flat_spike_input(const char *filename, size_t total_size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open spike input file");
        return NULL;
    }

    float *input = (float *)malloc(total_size * sizeof(float));
    if (!input) {
        perror("Failed to allocate memory for input");
        fclose(file);
        return NULL;
    }

    for (size_t i = 0; i < total_size; ++i) {
        if (fscanf(file, "%f", &input[i]) != 1) {
            fprintf(stderr, "Error reading float at index %zu\n", i);
            free(input);
            fclose(file);
            return NULL;
        }
    }

    fclose(file);
    return input;
}


// Function to convert NMNIST events to discretized input
float *convert_events_to_input(const NMNISTEvent *events, size_t num_events, int time_bins, int height, int width, unsigned int max_time) {
    // Allocate a 4D array: [2 (ON/OFF)][T][H][W]
    size_t input_size = 2 * time_bins * height * width; // 2 channels (ON/OFF)
    float *input = (float *)calloc(input_size, sizeof(float)); // Initialize to 0
    if (!input) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    unsigned int bin_size = max_time / time_bins;

    // Populate the bins
    for (size_t i = 0; i < num_events; i++) {
        const NMNISTEvent *event = &events[i];

        // Determine the time bin
        int t = event->timestamp / bin_size;
        if (t >= time_bins) continue; // Ignore events outside the time range

        // Map (x, y) to the spatial dimensions (0-based indexing)
        int y = event->y; 
        int x = event->x; 
        if (x < 0 || x >= width || y < 0 || y >= height) continue; 

        // Determine the channel (0 for OFF, 1 for ON)
        int c = event->polarity;  

        // Compute the flattened index
        // [C][T][X][Y]
        size_t index = (c * time_bins * height * width) + (t * height * width) + (y * width) + x;

        // Accumulate spikes in the appropriate channel
        input[index] += 1.0f;
    }

    return input; // Caller must free the memory
}

void plot_event_grid(NMNISTSample *sample, int axis_x, int axis_y, int plot_frame_number) {
    FILE *gnuplotPipe = popen("gnuplot -persistent", "w");
    if (!gnuplotPipe) {
        fprintf(stderr, "Error: Could not open gnuplot.\n");
        return;
    }

    // Configure Gnuplot
    fprintf(gnuplotPipe, "set multiplot layout %d,%d title 'Event Grid Heatmap'\n", axis_x, axis_y);
    fprintf(gnuplotPipe, "unset key\n");
    fprintf(gnuplotPipe, "set size ratio -1\n");  // Keep correct aspect ratio
    fprintf(gnuplotPipe, "unset xtics\n");
    fprintf(gnuplotPipe, "unset ytics\n");
    fprintf(gnuplotPipe, "unset border\n");
    fprintf(gnuplotPipe, "set xrange [0:33]\n");
    fprintf(gnuplotPipe, "set yrange [33:0]\n");  // Flip Y-axis
    fprintf(gnuplotPipe, "set palette rgbformulae 30,31,32\n");  // Better heatmap color
    fprintf(gnuplotPipe, "set cbrange [0:*]\n");  // Auto color scaling based on event density

    // Calculate the number of frames
    int num_frames = axis_x * axis_y;
    int events_per_frame = sample->num_events / num_frames;

    for (int i = 0; i < num_frames; i++) {
        fprintf(gnuplotPipe, "set title 'Frame %d'\n", i);
        fprintf(gnuplotPipe, "plot '-' matrix with image\n");

        // Create a 2D event grid (initialize with zeros)
        int event_grid[34][34] = {0};

        // Count events per (x, y) location
        for (int j = 0; j < events_per_frame; j++) {
            int index = i * events_per_frame + j;
            if (index < sample->num_events) {
                NMNISTEvent event = sample->events[index];
                event_grid[event.y][event.x]++;  // Swap X/Y if needed
            }
        }

        // Output the grid data in Gnuplot's matrix format
        for (int y = 0; y < 34; y++) {
            for (int x = 0; x < 34; x++) {
                fprintf(gnuplotPipe, "%d ", event_grid[y][x]);
            }
            fprintf(gnuplotPipe, "\n");
        }
        fprintf(gnuplotPipe, "e\n");
    }

    fprintf(gnuplotPipe, "unset multiplot\n");
    fflush(gnuplotPipe);
    pclose(gnuplotPipe);
}

// Save a single 2D frame as a PGM image
void save_frame_as_pgm(const char *filename, float *data, int height, int width) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Error: Could not open file %s for writing\n", filename);
        return;
    }

    // Write PGM header
    fprintf(file, "P2\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "255\n");

    // Normalize and write pixel data
    float max_value = 0.0f;
    for (int i = 0; i < height * width; i++) {
        if (data[i] > max_value) max_value = data[i];
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;
            int value = (int)((data[index] / max_value) * 255); // Normalize to 0-255
            fprintf(file, "%d ", value);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    printf("Saved frame to %s\n", filename);
}

// Save a single 2D frame as a PNG or JPG image
void save_frame_as_image(const char *filename, float *data, int height, int width, int is_png) {
    // Allocate memory for 8-bit grayscale image
    uint8_t *image = (uint8_t *)malloc(height * width);
    if (!image) {
        printf("Error: Could not allocate memory for image\n");
        return;
    }

    // Find the max value in data for normalization
    float max_value = 0.0f;
    for (int i = 0; i < height * width; i++) {
        if (data[i] > max_value) max_value = data[i];
    }

    // Normalize and convert to 8-bit grayscale
    for (int i = 0; i < height * width; i++) {
        image[i] = (uint8_t)((data[i] / max_value) * 255);
    }

    // Save as PNG or JPG
    int success;
    if (is_png) {
        success = stbi_write_png(filename, width, height, 1, image, width);
    } else {
        success = stbi_write_jpg(filename, width, height, 1, image, 90); // 90% quality for JPG
    }

    // Cleanup
    free(image);

    if (success) {
        printf("Saved frame to %s\n", filename);
    } else {
        printf("Error: Could not save image %s\n", filename);
    }
}

// Print a 28x28 array of floats with 1-digit precision
void print_frame(float *data, int height, int width) {
    if (height != 34 || width != 34) {
        printf("Error: Expected a 28x28 frame, but got %dx%d\n", height, width);
        return;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%.1f ", data[y * width + x]);
        }
        printf("\n");
    }
}

// Generate and save temporal frames for a sample
void visualize_sample_frames(float *discretized_input, const char *output_dir, int time_bins, int height, int width, unsigned int max_time) {
    // Convert events to discretized input
    //float *discretized_input = convert_events_to_input(
    //    sample->events, sample->num_events, time_bins, height, width, max_time);

    // Create output directory if it doesn't exist
    char mkdir_command[256];
    snprintf(mkdir_command, sizeof(mkdir_command), "mkdir -p %s", output_dir);
    system(mkdir_command);

    // Save each time bin as a separate frame
    for (int t = 0; t < time_bins; t++) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/frame_%03d.pgm", output_dir, t);

        // Extract the 2D frame for the current time bin
        float *frame = &discretized_input[t * height * width];
        save_frame_as_pgm(filename, frame, height, width);
    }

    free(discretized_input);
}

// Free the NMNIST dataset
void free_nmnist_dataset(NMNISTDataset *dataset) {
    for (size_t i = 0; i < dataset->num_samples; i++) {
        free(dataset->samples[i].events);
    }
    free(dataset->samples);
    free(dataset);
}
