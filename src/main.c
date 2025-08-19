#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stdlib.h"
#include <sys/time.h>

#include "../include/tests.h"
#include "../include/models/lif_neuron.h"
#include "../include/models/izhikevich_neuron.h"
#include "../include/utils/perf.h"

static double now_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
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
	nmnist_test();
	//stmnist_test();

	double t1 = now_seconds();
	printf("train_test() took %.6f seconds\n", t1 - t0);

	if (perf_mode) perf_report();

	//iris_classification_example();
	//hor_vert_dataset();

	return 0;
}

