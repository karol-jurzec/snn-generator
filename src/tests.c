#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "snn.c"
#include "stdlib.h"
#include "../utils/snn_plot.c"

void single_lf_test(ModelBase* model_base) {
    int dt = 200; 
    FILE *log_file = fopen("out/single_neuron_output.txt", "w");
        if (log_file == NULL) {
            perror("Error opening log file");
            return;
        }

    double arr[dt];
    for (int i = 0; i < 200; ++i) {
        if (i < 100) {
            arr[i] = 0.0;
        } else if (i < 150) {
            arr[i] = 0.25;
        } else {
            arr[i] = 0.0;
        }
        }

    for (int i = 0; i < dt; ++i) {
        model_base->update_neuron(model_base, arr[i]);
        log_membrane_potential(log_file, model_base->v, i);
    }

    fclose(log_file);
    plot_single_neuron("out/single_neuron_output.txt", "out/single_neuron_plot.png");
}