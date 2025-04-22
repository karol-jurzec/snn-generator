#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stdlib.h"

#include "../include/tests.h"
#include "../include/models/lif_neuron.h"
#include "../include/models/izhikevich_neuron.h"

int main(int argc, char *argv[]) {

    /* 
    float **array = (float **)malloc(4 * sizeof(float *));
    for (int i = 0; i < 4; i++) {
        array[i] = (float *)malloc(4 * sizeof(float));
    }

    array[0][0] = 0.2; array[0][1] = 0.5; array[0][2] = 0.7; array[0][3] = 0.1;
    array[1][0] = 0.6; array[1][1] = 0.3; array[1][2] = 0.9; array[1][3] = 0.4;
    array[2][0] = 0.8; array[2][1] = 0.2; array[2][2] = 0.5; array[2][3] = 0.7;
    array[3][0] = 0.3; array[3][1] = 0.6; array[3][2] = 0.8; array[3][3] = 0.2;
 

    SNN network;
    initialize_snn(&network); 
    update_snn(&network,  array, 4);

    get_snn_spike_counts(&network);

    LIFNeuron* leaky = malloc(sizeof(LIFNeuron));

    initialize_neuron(leaky, 0.0, 1.0, 0.0, 0.8);
    single_lf_test(&(leaky->model_base));
    free(leaky); // Free allocated memory
    */ 
   
   /*
    LIFNeuron lif;
    lif_initialize(&lif, 0.0f, 1.0f, 0.0f, 0.8f);
    single_neuron_test((ModelBase *)&lif, "out/single_neuron_leaky.png");

    IzhikevichNeuron izh;
    izhikevich_initialize(&izh, 0.1f, 0.1f, 0.1f, 0.1f, 0.0f);
    single_neuron_test((ModelBase *)&izh, "out/single_neuron_izh.png");

    conv2d_test();
    maxpool2d_test();
    flatten_test();
    linear_test();
    spiking_layer_test();
    network_test();
    network_loader_test();
   */

    nmnist_loader_test();

    return 0;
}
