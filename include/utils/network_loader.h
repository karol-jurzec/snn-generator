#ifndef NETWORK_LOADER_H
#define NETWORK_LOADER_H

#include <json-c/json.h>
#include "../network.h"
#include "../models/model_base.h"

// Function to load and initialize a network from a configuration file
Network *initialize_network_from_file(const char *config_path);
void parse_and_add_layer(Network *network, struct json_object *layer_config, size_t index);
ModelBase **initialize_neurons(const char *neuron_type, size_t num_neurons, struct json_object *neuron_config);

#endif // NETWORK_LOADER_H
