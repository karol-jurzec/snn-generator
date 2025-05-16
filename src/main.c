#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include "../include/utils/events.h"
#include "../include/utils/network_loader.h"
#include "../include/network.h"

#define GRID_SIZE 10
#define CELL_SIZE 20
#define WINDOW_SIZE (GRID_SIZE * CELL_SIZE)

int event_counter = 0;

#define CTRL_WIDTH 200
#define CTRL_HEIGHT WINDOW_SIZE

int grid[GRID_SIZE][GRID_SIZE] = {0};
Uint32 cell_activation_time[GRID_SIZE][GRID_SIZE] = {0};
Uint32 cell_deactivation_time[GRID_SIZE][GRID_SIZE] = {0};
bool cells_currently_touched[GRID_SIZE][GRID_SIZE] = {0};

#define SPIKE_DURATION 80
#define BRUSH_RADIUS 0.5

bool cell_previous_state[GRID_SIZE][GRID_SIZE] = {0};

#define FRAME_WINDOW_US 20000  // 20 ms


#define MAX_EVENTS_PER_FRAME 100
SpikeEvent frame_events[MAX_EVENTS_PER_FRAME];
int frame_event_count = 0;
int no_frame = 0;

Uint64 performance_frequency;
Uint64 get_time_us() {
    return (SDL_GetPerformanceCounter() * 1000000ULL) / performance_frequency;
}

void generate_frame_events(Uint64 frame_start_us, Uint32 current_ticks) {
    frame_event_count = 0;

    for (int y = 0; y < GRID_SIZE; ++y) {
        for (int x = 0; x < GRID_SIZE; ++x) {
            bool current = grid[y][x];
            bool previous = cell_previous_state[y][x];

            if (current != previous && frame_event_count < MAX_EVENTS_PER_FRAME) {
                SpikeEvent ev;
                ev.timestamp = (unsigned int)(get_time_us() - frame_start_us);
                ev.x = x;
                ev.y = y;
                ev.neuron_id = y * GRID_SIZE + x;
                ev.polarity = current ? 0 : 1;  // 1 for ON, 0 for OFF
                ev.channel = ev.polarity;
                frame_events[frame_event_count++] = ev;

                event_counter++;

                // Update previous state immediately on state change
                cell_previous_state[y][x] = current;

                if (!current) {
                    cell_deactivation_time[y][x] = current_ticks;
                } else {
                    cell_activation_time[y][x] = current_ticks;
                }
            }
        }
    }
}



bool timer_running = false;
Uint32 start_time = 0;
Uint32 elapsed_time = 0;
bool mouse_down = false;

void format_time(Uint32 ms, char *buf, size_t buf_size) {
    Uint32 total_seconds = ms / 1000;
    Uint32 minutes = total_seconds / 60;
    Uint32 seconds = total_seconds % 60;
    snprintf(buf, buf_size, "%02u:%02u", minutes, seconds);
}

void activate_cells(int cx, int cy, float radius, Uint32 current_ticks) {
    // Clear previous touched cells first
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            cells_currently_touched[y][x] = false;
        }
    }

    int min_y = (int)fmaxf(0, floorf(cy - radius));
    int max_y = (int)fminf(GRID_SIZE - 1, ceilf(cy + radius));
    int min_x = (int)fmaxf(0, floorf(cx - radius));
    int max_x = (int)fminf(GRID_SIZE - 1, ceilf(cx + radius));

    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {
            float dx = x - cx;
            float dy = y - cy;
            if ((dx * dx + dy * dy) <= (radius * radius)) {
                grid[y][x] = 1;
                cell_activation_time[y][x] = current_ticks;
                cells_currently_touched[y][x] = true;
            }
        }
    }
}

void print_float_array(float *arr, int length) {
    printf("[");
    for (int i = 0; i < length; i++) {
        printf("%.2f", arr[i]);  // print with 2 decimal places
        if (i < length - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}


int predict(Network *network, float* input, unsigned int timestep) {

    forward(network, input, 10 * 10 * 2, timestep);

    SpikingLayer *output_layer = (SpikingLayer *)network->layers[network->num_layers - 1];
    float spike_counts[output_layer->num_neurons];
    for (size_t n = 0; n < output_layer->num_neurons; n++) {
        LIFNeuron *neuron = (LIFNeuron *)output_layer->neurons[n];
        spike_counts[n] = (float)neuron->spike_count;
        
    }

    print_float_array(spike_counts, output_layer->num_neurons);
}


void update_spikes(Uint32 current_ticks) {
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            if (cells_currently_touched[y][x]) {
                // While touched, keep active and refresh activation time
                cell_activation_time[y][x] = current_ticks;
                grid[y][x] = 1;
                cell_deactivation_time[y][x] = 0;  // reset OFF timer on reactivation
            } else if (grid[y][x] == 1) {
                // Not touched but active — check if spike duration passed
                if ((current_ticks - cell_activation_time[y][x]) > SPIKE_DURATION) {
                    grid[y][x] = 0;
                    cell_activation_time[y][x] = 0;
                    cell_deactivation_time[y][x] = current_ticks;  // record OFF start time
                }
            } else {
                // If cell is OFF, clear activation time if spike duration passed for OFF events (optional)
                if (cell_deactivation_time[y][x] != 0 && (current_ticks - cell_deactivation_time[y][x]) > SPIKE_DURATION) {
                    cell_deactivation_time[y][x] = 0;
                }
            }
        }
    }
}



void draw_grid(SDL_Renderer *renderer) {
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            SDL_Rect cell = {x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE};
            SDL_SetRenderDrawColor(renderer, grid[y][x] ? 0 : 255, grid[y][x] ? 0 : 255, grid[y][x] ? 0 : 255, 255);
            SDL_RenderFillRect(renderer, &cell);
            SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
            SDL_RenderDrawRect(renderer, &cell);
        }
    }
}

void clear_grid() {
    start_time = 0; 
    elapsed_time = 0;
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            grid[y][x] = 0;
            cell_activation_time[y][x] = 0;
            cell_previous_state[y][x] = 0;
        }
    }
}

void draw_controls(SDL_Renderer *renderer, Uint32 elapsed_ms) {
    SDL_SetRenderDrawColor(renderer, 220, 220, 220, 255);
    SDL_RenderClear(renderer);

    SDL_Rect timer_rect = {20, 20, CTRL_WIDTH - 40, 50};
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderFillRect(renderer, &timer_rect);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderDrawRect(renderer, &timer_rect);

    SDL_Rect reset_button = {20, 100, CTRL_WIDTH - 40, 50};
    SDL_SetRenderDrawColor(renderer, 180, 0, 0, 255);
    SDL_RenderFillRect(renderer, &reset_button);
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderDrawRect(renderer, &reset_button);
}

bool point_in_rect(int px, int py, SDL_Rect *rect) {
    return px >= rect->x && px < (rect->x + rect->w) && py >= rect->y && py < (rect->y + rect->h);
}

int main(int argc, char *argv[]) {
    const char *model_architecrure = "scnn_stmnist_architecture.json";
    const char *model_weights = "scnn_stmnist_weights_bs_64.json";
    const char *dataset_path_test = "C:/Users/karol/Desktop/karol/agh/praca_snn/dataset/STMNIST/data_submission"; 

    printf("Loading network from %s...\n", model_architecrure);
    Network *network = initialize_network_from_file(model_architecrure, 10, 10, 2);
    if (!network) {
        printf("Error: Failed to load network.\n");
        return 0;
    }

    load_weights_from_json(network, model_weights);
    printf("Weights were read succesfully...\n");

    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER);

    performance_frequency = SDL_GetPerformanceFrequency();

    SDL_Window *grid_window = SDL_CreateWindow("ST-MNIST Simulator", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WINDOW_SIZE, WINDOW_SIZE, SDL_WINDOW_SHOWN);
    SDL_Renderer *grid_renderer = SDL_CreateRenderer(grid_window, -1, SDL_RENDERER_ACCELERATED);

    //SDL_Window *ctrl_window = SDL_CreateWindow("Controls", SDL_WINDOWPOS_CENTERED + WINDOW_SIZE + 20, SDL_WINDOWPOS_CENTERED, CTRL_WIDTH, CTRL_HEIGHT, SDL_WINDOW_SHOWN);
    //SDL_Renderer *ctrl_renderer = SDL_CreateRenderer(ctrl_window, -1, SDL_RENDERER_ACCELERATED);

    bool running = true;
    SDL_Event event;
    SDL_Rect reset_button = {20, 100, CTRL_WIDTH - 40, 50};

    Uint64 last_frame_time_us = get_time_us();

    while (running) {
        Uint32 current_ticks = SDL_GetTicks();
        Uint64 current_time_us = get_time_us();

        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    running = false;
                    break;
                case SDL_MOUSEBUTTONDOWN:
                    if (event.button.windowID == SDL_GetWindowID(grid_window) && event.button.button == SDL_BUTTON_LEFT) {
                        mouse_down = true;
                        int mx = event.button.x / CELL_SIZE;
                        int my = event.button.y / CELL_SIZE;
                        activate_cells(mx, my, BRUSH_RADIUS, current_ticks);
                        if (!timer_running) {
                            start_time = current_ticks;
                            timer_running = true;
                        }
                    } 
                    // else if (event.button.windowID == SDL_GetWindowID(ctrl_window)) {
                    //     int cx = event.button.x;
                    //     int cy = event.button.y;
                    //     if (point_in_rect(cx, cy, &reset_button)) {
                    //         clear_grid();
                    //         timer_running = false;
                    //         elapsed_time = 0;
                    //     }
                    // }
                    break;
                case SDL_MOUSEBUTTONUP:
                    if (event.button.windowID == SDL_GetWindowID(grid_window) && event.button.button == SDL_BUTTON_LEFT) {
                        mouse_down = false;
                    
                        // Clear touched cells on mouse release
                        for (int y = 0; y < GRID_SIZE; y++) {
                            for (int x = 0; x < GRID_SIZE; x++) {
                                cells_currently_touched[y][x] = false;
                            }
                        }
                    }
                    break;
                case SDL_MOUSEMOTION:
                    if (mouse_down && event.motion.windowID == SDL_GetWindowID(grid_window)) {
                        int mx = event.motion.x / CELL_SIZE;
                        int my = event.motion.y / CELL_SIZE;
                        activate_cells(mx, my, BRUSH_RADIUS, current_ticks);
                    }
                    break;
                case SDL_KEYDOWN:
                    if (event.key.keysym.sym == SDLK_c) {
                        clear_grid();
                        timer_running = false;
                        elapsed_time = 0;
                    }
                    break;
            }
        }

        if (timer_running) {
            elapsed_time = current_ticks - start_time;
        }

        update_spikes(current_ticks);
        if (current_time_us - last_frame_time_us >= FRAME_WINDOW_US / 2) {
            generate_frame_events(last_frame_time_us, current_ticks);
            
        }

        if (current_time_us - last_frame_time_us >= FRAME_WINDOW_US && frame_event_count > 0) {
            no_frame++;
            last_frame_time_us = current_time_us;
            
            printf("frame event count %d\n", frame_event_count);

            int16_t ****single_frame = events_to_single_frame(frame_events, frame_event_count, GRID_SIZE, GRID_SIZE);

            if (single_frame) {
                
                float* input = flatten_frames_to_float(single_frame, 1, 10, 10);
                predict(network, input, current_time_us / 1000);
                free_frames(single_frame, 1, GRID_SIZE);
            }


            frame_event_count = 0;

        }

        SDL_SetRenderDrawColor(grid_renderer, 255, 255, 255, 255);
        SDL_RenderClear(grid_renderer);
        draw_grid(grid_renderer);
        SDL_RenderPresent(grid_renderer);

        //draw_controls(ctrl_renderer, elapsed_time);
        //SDL_RenderPresent(ctrl_renderer);

        // Uint64 loop_duration_us = get_time_us() - current_time_us;
        // if (loop_duration_us < 50000) {
        //     SDL_Delay((50000 - loop_duration_us) / 1000);
        // }

        SDL_Delay(0);
    }

    SDL_DestroyRenderer(grid_renderer);
    SDL_DestroyWindow(grid_window);
    //SDL_DestroyRenderer(ctrl_renderer);
    //SDL_DestroyWindow(ctrl_window);
    SDL_Quit();

    return 0;
}
