#include "../../include/utils/dataset_loader.h"

// Generic function to load a dataset based on the format
Dataset *load_dataset(const char *source_path, DatasetFormat format, size_t max_samples, bool stabilize,  bool denoise) {
    switch (format) {
        case FORMAT_NMNIST:
            return load_nmnist_dataset(source_path, max_samples, stabilize, denoise);  // This can be adapted for other formats
        // Add additional cases for other dataset formats here (e.g., NCARS, HDF5_SHD, etc.)
        default:
            //fprintf(stderr, "Unsupported dataset format.\n");
            //return NULL;
    }
}
