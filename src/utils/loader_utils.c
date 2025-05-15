#include "../../include/utils/loader_utils.h"

void shuffle_dataset(Dataset *dataset) {
    for (size_t i = 0; i < dataset->num_samples; i++) {
        size_t j = i + rand() % (dataset->num_samples - i);
        
        // Swap samples
        Sample temp = dataset->samples[i];
        dataset->samples[i] = dataset->samples[j];
        dataset->samples[j] = temp;
    }
}