#ifndef STMNIST_LOADER_H
#define STMNIST_LOADER_H

#include <matio.h>
#include <dirent.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>

#include "dataset_loader.h"
#include "events.h"
#include "loader_utils.h"

Dataset *load_stmnist_dataset(const char *dir, size_t max_samples, bool stabilize, bool denoise);

#define STMNIST_WIDTH 10
#define STMNIST_HEIGHT 10
#define STMNIST_CLASSES 10
#define STMNIST_CHANNELS 2 

#endif // STMNIST_LOADER_H