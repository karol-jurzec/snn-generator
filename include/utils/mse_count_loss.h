#include <math.h>
#include <stdlib.h>

typedef struct {
    float* spike_counts;       // Batch x num_neurons
    float* target_counts;      // Batch x num_neurons
    int batch_size;
    int num_neurons;
    int num_steps;
    float correct_rate;
    float incorrect_rate;
} MSECountLoss;

void init_mse_count_loss(MSECountLoss* loss, int batch_size, int num_neurons, int num_steps, 
                         float correct_rate, float incorrect_rate);
void free_mse_count_loss(MSECountLoss* loss);
void generate_target_counts(MSECountLoss* loss, int* labels);
float compute_mse_loss(MSECountLoss* loss);