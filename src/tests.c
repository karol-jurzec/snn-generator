#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

#include "../include/tests.h"
#include "../include/network.h"

#include "../include/models/model_base.h"
#include "../include/models/lif_neuron.h"

#include "../include/utils/snn_plot.h"
#include "../include/utils/layer_utils.h"
#include "../include/utils/network_loader.h"
#include "../include/utils/network_logger.h"
#include "../include/utils/nmnist_loader.h"
#include "../include/utils/dataset_loader.h"
#include "../include/utils/perf.h"

#include "../include/layers/conv2d_layer.h"
#include "../include/layers/maxpool2d_layer.h"
#include "../include/layers/flatten_layer.h"
#include "../include/layers/linear_layer.h"
#include "../include/layers/spiking_layer.h"

static double now_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void stmnist_test() {
	const char *model_architecrure = "scnn_stmnist_architecture.json";
	const char *model_weights = "scnn_stmnist_weights_bs_64.json";
	const char *dataset_path_test = getenv("STMNIST_TEST_PATH");

    if (!dataset_path_test) {
		fprintf(stderr, "Error: STMNIST_TEST_PATH environment variable not set\n");
		return;
	}

	printf("Loading network from %s...\n", model_architecrure);
	perf_mark_start("network_init");
	Network *network = initialize_network_from_file(model_architecrure, 10, 10, 2);
	perf_mark_end("network_init");
	if (!network) {
		printf("Error: Failed to load network.\n");
		return;
	}

	perf_mark_start("weights_load");
	load_weights_from_json(network, model_weights);
	perf_mark_end("weights_load");
	printf("Weights were read succesfully...\n");

	printf("Loading test dataset from %s...\n", dataset_path_test);
	perf_mark_start("dataset_load");
	Dataset *dataset_test = load_dataset(dataset_path_test, FORMAT_STMNIST, 10, false, false);
	perf_mark_end("dataset_load");
	if (dataset_test) {
		perf_add_metric("dataset_samples", (double)dataset_test->num_samples);
	}

	printf("Testing the network accuracy...\n");

	double t0 = now_seconds();

	perf_mark_start("inference_total");
	test(network, dataset_test);
	perf_mark_end("inference_total");

	double t1 = now_seconds();
	printf("train_test() took %.6f seconds\n", t1 - t0);

	optimize_network(network);

	Dataset *dataset_test_v2 = load_dataset(dataset_path_test, FORMAT_STMNIST, 100, false, false);

	perf_mark_start("inference_total");
	test(network, dataset_test_v2);
	perf_mark_end("inference_total");

	free_network(network);

	printf("Test was completed successfully.\n");
}

void nmnist_test() {
    const char *network_config_path_train = getenv("NMNIST_TRAIN_PATH");
    const char *network_config_path_test = getenv("NMNIST_TEST_PATH");
    const char *dataset_path = "snn_nmnist_architecture.json";

    // Load the network
    printf("Loading network from %s...\n", dataset_path);
    perf_mark_start("nmnist_network_init");
    Network *network = initialize_network_from_file(dataset_path, 34, 34, 2);
    perf_mark_end("nmnist_network_init");
    if (!network) {
        printf("Error: Failed to load network.\n");
        return;
    }

    perf_mark_start("nmnist_weights_load");
    load_weights_from_json(network, "snn_nmnist_weights_bs_32.json");
    perf_mark_end("nmnist_weights_load");

    printf("Loading test dataset from %s...\n", network_config_path_test);
    perf_mark_start("nmnist_dataset_load");
    Dataset *dataset_test = load_dataset(network_config_path_test, FORMAT_NMNIST, 1, false, true);
    perf_mark_end("nmnist_dataset_load");
    if (dataset_test) {
        perf_add_metric("nmnist_dataset_samples", (double)dataset_test->num_samples);
    }

    printf("Testing the network accuracy...\n");
    perf_mark_start("nmnist_inference_total");
    float acc = test(network, dataset_test);  // Call after training
    free_dataset(dataset_test);
    perf_mark_end("nmnist_inference_total");
    perf_add_metric("nmnist_accuracy_percent", (double)acc);

    // Clean up
    //_dataset(dataset_test);
    //free_nmnist_dataset(dataset_train);
    free_network(network);

    printf("Training test completed successfully.\n");
}






