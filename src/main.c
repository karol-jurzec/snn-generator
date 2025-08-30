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
    test_channel_pruning(
        "snn_nmnist_architecture.json",           // architektura
        "snn_nmnist_weights_bs_32.json",          // wagi
        "C:/Users/karol/Desktop/karol/agh/praca_snn/data/NMNIST/Test", // dataset
        100,                                        // liczba próbek do analizy
        1000,                                        // threshold spike-ów (0 = wszystkie bez spike-ów)
		FORMAT_NMNIST,
		34, 34, 2
	);
}

void test_stmnist_pruning() {
    test_channel_pruning(
        "scnn_stmnist_architecture.json",           // architektura
        "scnn_stmnist_weights_bs_64.json",          // wagi
        "C:/Users/karol/Desktop/karol/agh/praca_snn/dataset/STMNIST/data_submission", // dataset
        250,                                        // liczba próbek do analizy
        0,                                         // threshold spike-ów (0 = wszystkie bez spike-ów)
		FORMAT_STMNIST,
		10, 10, 2
	);
}

void study_stmnist_threshold_impact() {
    study_threshold_impact(
        "scnn_stmnist_architecture.json",           // architektura
        "scnn_stmnist_weights_bs_64.json",          // wagi  
        "C:/Users/karol/Desktop/karol/agh/praca_snn/dataset/STMNIST/data_submission", // dataset
        "stmnist_pruning_threshold_study.csv",      // plik wyników
        250                                         // liczba próbek (250 do analizy + 250 do testów)
    );
}

void test_stmnist_bidirectional_pruning() {
    test_bidirectional_pruning(
        "scnn_stmnist_architecture.json",           
        "scnn_stmnist_weights_bs_64.json",          
        "C:/Users/karol/Desktop/karol/agh/praca_snn/dataset/STMNIST/data_submission", 
        250,                                      // więcej próbek dla wiarygodności
        0,                                        // threshold
        FORMAT_STMNIST,
        10, 10, 2
    );
}

int main(int argc, char *argv[]) {
	
	//nmnist_loader_test();
	srand(42);

	int perf_mode = 0;
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--perf") == 0) perf_mode = 1;		
	}


	if (perf_mode) perf_enable(1);

	double t0 = now_seconds();

	//loader_test();
	//nmnist_test();
	//stmnist_test();

    //study_stmnist_threshold_impact();

	//test_nmnist_bidirectional_pruning();
	//test_stmnist_pruning();
	test_stmnist_bidirectional_pruning();

	double t1 = now_seconds();
	printf("train_test() took %.6f seconds\n", t1 - t0);

	if (perf_mode) perf_report();

	//iris_classification_example();
	//hor_vert_dataset();

	return 0;
}

