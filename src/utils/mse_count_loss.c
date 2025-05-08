#include "../../include/utils/mse_count_loss.h"

void init_mse_count_loss(MSECountLoss* loss, int batch_size, int num_neurons, int num_steps, 
                         float correct_rate, float incorrect_rate) {
    loss->batch_size = batch_size;
    loss->num_neurons = num_neurons;
    loss->num_steps = num_steps;
    loss->correct_rate = correct_rate;
    loss->incorrect_rate = incorrect_rate;
    
    loss->spike_counts = (float*)malloc(batch_size * num_neurons * sizeof(float));
    loss->target_counts = (float*)malloc(batch_size * num_neurons * sizeof(float));
}

void free_mse_count_loss(MSECountLoss* loss) {
    free(loss->spike_counts);
    free(loss->target_counts);
}

void generate_target_counts(MSECountLoss* loss, int* labels) {
    float correct_target = loss->num_steps * loss->correct_rate; 
    float incorrect_target = loss->num_steps * loss->incorrect_rate;  
    
    for (int b = 0; b < loss->batch_size; b++) {
        for (int n = 0; n < loss->num_neurons; n++) {
            if (n == labels[b]) {
                loss->target_counts[b * loss->num_neurons + n] = correct_target;
            } else {
                loss->target_counts[b * loss->num_neurons + n] = incorrect_target;
            }
        }
    }
}

float compute_mse_loss(MSECountLoss* loss) {
    float total_loss = 0.0f;
    int total_elements = loss->batch_size * loss->num_neurons;
    
    for (int i = 0; i < total_elements; i++) {
        float diff = (loss->spike_counts[i] - loss->target_counts[i]) / loss->num_steps;
        total_loss += diff * diff;
    }
    
    return total_loss / total_elements;  // Remove division by num_steps here
}
