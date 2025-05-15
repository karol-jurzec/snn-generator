#ifndef NMNIST_LOADER_H
#define NMNIST_LOADER_H

#include "events.h"
#include "dataset_loader.h"
#include "loader_utils.h"

// NMNIST-specific functions
Dataset *load_nmnist_dataset(const char *dir, size_t max_samples, bool stabilize, bool denoise);

// NMNIST constants
#define NMNIST_CHANNELS 2 
#define NMNIST_WIDTH 34
#define NMNIST_HEIGHT 34
#define NMNIST_CLASSES 10

#endif // NMNIST_LOADER_H