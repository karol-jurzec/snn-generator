#ifndef NETWORK_LOADER_H
#define NETWORK_LOADER_H

#include <json-c/json.h>
#include "../network.h"
#include "../models/model_base.h"

Network *initialize_network_from_file(const char *config_path, int input_heigth, int input_width, int input_channels);
void parse_and_add_layer(Network *network, struct json_object *layer_config, size_t index);
ModelBase **initialize_neurons(const char *neuron_type, size_t num_neurons, struct json_object *neuron_config);
void load_weights_from_json(Network *network, const char *weights_path);

#endif // NETWORK_LOADER_H
