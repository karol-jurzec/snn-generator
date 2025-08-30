#include "../../include/utils/network_logger.h"
#include "../../include/layers/spiking_layer.h"

void create_directory(const char *path);

void log_weights(Network *network, int epoch, int batch) {
    char filename[50];
    snprintf(filename, sizeof(filename), "out/weights/weights_epoch_%d_batch_%d.txt", epoch, batch);

    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening weight log file!\n");
        return;
    }

    fprintf(file, "Epoch %d\n", epoch);
    for (size_t l = 0; l < network->num_layers; l++) {
        LayerBase *layer = network->layers[l];
        if (layer->weights && layer->num_weights > 0) {
            fprintf(file, "Layer %zu\n", l);
            for (size_t w = 0; w < layer->num_weights; w++) {
                fprintf(file, "%f\n", layer->weights[w]);
            }
        }
    }
    fclose(file);
}

const char* get_layer_type_name(LayerType type) {
    switch (type) {
        case LAYER_CONV2D: return "Conv2D";
        case LAYER_LINEAR: return "Linear";
        case LAYER_MAXPOOL2D: return "MaxPool2D";
        case LAYER_FLATTEN: return "Flatten";
        case LAYER_SPIKING: return "Spiking";
        default: return "Unknown";
    }
}

void log_inputs(Network *network, int epoch, int sample, int t) {
    const char *base_dir = "out/inputs";

    for (size_t l = 0; l < network->num_layers; l++) {
        LayerBase *layer = network->layers[l];

        if (layer->inputs && layer->num_inputs > 0) {
            const char *type_str = get_layer_type_name(layer->layer_type);

            // Construct subdirectory path: e.g., out/inputs/Conv2D_0/
            char dir_path[256];
            snprintf(dir_path, sizeof(dir_path), "%s/%s_%zu", base_dir, type_str, l);

            // Create directory if it doesn't exist
            create_directory(base_dir);
            create_directory(dir_path);

            // Build full file path
            char filepath[300];
            snprintf(filepath, sizeof(filepath), "%s/inputs_epoch_%d_sample_%d_t_%d.txt", dir_path, epoch, sample, t);

            FILE *file = fopen(filepath, "w");
            if (!file) {
                perror("Failed to open input log file");
                continue;
            }

            fprintf(file, "Epoch %d, Sample %d, Time %d\n", epoch, sample, t);
            fprintf(file, "Layer %zu (%s)\n", l, type_str);
            for (size_t i = 0; i < layer->num_inputs; i++) {
                fprintf(file, "%f\n", layer->inputs[i]);
            }
            fclose(file);
        }
    }
}

void log_outputs(Network *network, int epoch, int sample, int t) {
    const char *base_output_dir = "out/outputs";
    create_directory(base_output_dir);  // Ensure base directory exists

    for (size_t l = 0; l < network->num_layers; l++) {
        LayerBase *layer = network->layers[l];

        if (!layer || !layer->output || layer->output_size == 0) {
            continue;
        }

        const char *type_str = get_layer_type_name(layer->layer_type);

        // Use layer->type directly as folder name
        char layer_type_dir[100];
        snprintf(layer_type_dir, sizeof(layer_type_dir), "%s/%s_%zu", base_output_dir, type_str, l);
        create_directory(layer_type_dir);

        // Compose output file path
        char filename[150];
        snprintf(filename, sizeof(filename),
                 "%s/%s_epoch_%d_sample_%d_t_%d.txt",
                 layer_type_dir, type_str, epoch, sample, t);

        FILE *file = fopen(filename, "w");
        if (!file) {
            perror("Failed to open output file");
            continue;
        }

        fprintf(file, "Epoch %d, Sample %d, Time %d\n", epoch, sample, t);
        fprintf(file, "Layer %zu (%s)\n", l, type_str);

        for (size_t i = 0; i < layer->output_size; i++) {
            fprintf(file, "%f\n", layer->output[i]);
        }

        fclose(file);
    }
}



// 3 spiking layers 

// 16 time steps - x axis
// num_neurpons - y axis 

void create_directory(const char *path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        mkdir(path);
    }
}

void log_membranes(Network *network, int epoch, int sample, int t) {
    char base_path[100];
    snprintf(base_path, sizeof(base_path), "out/membrane_values/sample_%02d_epoch_%02d", sample, epoch);
    create_directory("out/membrane_values");
    create_directory(base_path);

    for (size_t l = 0; l < network->num_layers; l++) {
        LayerBase *layer = network->layers[l];
        if (layer->is_spiking && layer->inputs && layer->num_inputs > 0) {
            char layer_path[120];
            snprintf(layer_path, sizeof(layer_path), "%s/layer_%02zu", base_path, l);
            create_directory(layer_path);

            char filename[140];
            snprintf(filename, sizeof(filename), "%s/t_%02d.txt", layer_path, t);
            FILE *file = fopen(filename, "w");
            if (!file) continue;

            fprintf(file, "Epoch %d, Sample %d, Time %d\n", epoch, sample, t);
            fprintf(file, "Layer %zu\n", l);

            SpikingLayer *spiking_layer = (SpikingLayer *)layer;
            for (size_t i = 0; i < spiking_layer->num_neurons; i++) {
                fprintf(file, "%f\n", spiking_layer->neurons[i]->v);
            }
            fclose(file);
        }
    }


}

void log_spikes(Network *network, int epoch, int sample, int t, int label) {
    char base_path[100];
    snprintf(base_path, sizeof(base_path), "out/spikes_outputs/sample_%02d_epoch_%02d_label_%02d", sample, epoch, label);
    create_directory("out/spikes_outputs");
    create_directory(base_path);

    for (size_t l = 0; l < network->num_layers; l++) {
        LayerBase *layer = network->layers[l];
        if (layer->is_spiking && layer->inputs && layer->num_inputs > 0) {
            char layer_path[120];
            snprintf(layer_path, sizeof(layer_path), "%s/layer_%02zu", base_path, l);
            create_directory(layer_path);

            char filename[140];
            snprintf(filename, sizeof(filename), "%s/t_%02d.txt", layer_path, t);
            FILE *file = fopen(filename, "w");
            if (!file) continue;

            fprintf(file, "Epoch %d, Sample %d, Time %d\n", epoch, sample, t);
            fprintf(file, "Layer %zu\n", l);
            for (size_t i = 0; i < layer->num_inputs; i++) {
                fprintf(file, "%f\n", layer->output[i]);
            }
            fclose(file);
        }
    }
}
