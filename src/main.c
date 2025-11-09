#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stdlib.h"
#include <sys/time.h>

#include "../include/tests.h"
#include "../include/models/lif_neuron.h"
#include "../include/models/izhikevich_neuron.h"
#include "../include/utils/perf.h"
#include "../include/utils/channel_pruning.h"
#include "../include/utils/bidirectional_pruning.h"

static double now_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void test_nmnist_pruning() {
    const char *dataset_path = getenv("NMNIST_TEST_PATH");
    if (!dataset_path) {
        fprintf(stderr, "Error: NMNIST_TEST_PATH environment variable not set\n");
        return;
    }
    
    test_channel_pruning(
        "snn_nmnist_architecture.json",
        "snn_nmnist_weights_bs_32.json", 
        dataset_path, 
        100,                                        
        0, // threshold value
		FORMAT_NMNIST,
		34, 34, 2
	);
}

void test_stmnist_pruning() {
    const char *dataset_path = getenv("STMNIST_TEST_PATH");
    if (!dataset_path) {
        fprintf(stderr, "Error: STMNIST_TEST_PATH environment variable not set\n");
        return;
    }
    
    test_channel_pruning(
        "scnn_stmnist_architecture.json",        
        "scnn_stmnist_weights_bs_64.json",     
        dataset_path,
        250,                                       
        0, // threshold value
		FORMAT_STMNIST,
		10, 10, 2
	);
}

void study_stmnist_threshold_impact() {
    const char *dataset_path = getenv("STMNIST_TEST_PATH");
    if (!dataset_path) {
        fprintf(stderr, "Error: STMNIST_TEST_PATH environment variable not set\n");
        return;
    }
    
    study_threshold_impact(
        "scnn_stmnist_architecture.json",          
        "scnn_stmnist_weights_bs_64.json",         
        dataset_path,
        "stmnist_pruning_threshold_study.csv",   
        250 
    );
}

void test_stmnist_bidirectional_pruning() {
    const char *dataset_path = getenv("STMNIST_TEST_PATH");
    if (!dataset_path) {
        fprintf(stderr, "Error: STMNIST_TEST_PATH environment variable not set\n");
        return;
    }
    
    test_bidirectional_pruning(
        "scnn_stmnist_architecture.json",           
        "scnn_stmnist_weights_bs_64.json",          
        dataset_path, 
        50,   
        0, // threshold
        FORMAT_STMNIST,
        10, 10, 2
    );
}

void test_nmnist_bidirectional_pruning() {
    const char *dataset_path = getenv("NMNIST_TEST_PATH");
    if (!dataset_path) {
        fprintf(stderr, "Error: NMNIST_TEST_PATH environment variable not set\n");
        return;
    }
    
    test_bidirectional_pruning(
        "snn_nmnist_architecture.json",          
        "snn_nmnist_weights_bs_32.json", 
        dataset_path, 
        10,                                       
        0,                                     
		FORMAT_NMNIST,
		34, 34, 2
    );
}


int main(int argc, char *argv[]) {
	
	srand(42);

	int perf_mode = 0;
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--perf") == 0) perf_mode = 1;		
	}


	if (perf_mode) perf_enable(1);

	double t0 = now_seconds();

	
	test_stmnist_bidirectional_pruning();

	//test_nmnist_bidirectional_pruning();

	double t1 = now_seconds();
	printf("train_test() took %.6f seconds\n", t1 - t0);

	if (perf_mode) perf_report();

	return 0;
}

