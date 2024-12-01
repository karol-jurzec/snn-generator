#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "snn.c"
#include "stdlib.h"

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
    */ 

   LIFNeuron leaky;
   initialize_neuron(&leaky, 0.0, 1.0, 0.0, 0.8);

   double arr[200];
   for (int i = 0; i < 200; ++i) {
       if (i < 100) {
           arr[i] = 0.0;
       } else if (i < 150) {
           arr[i] = 0.25;
       } else {
           arr[i] = 0.0;
       }
    }

    for (int i = 0; i < 200; ++i) {
        update_neuron(&leaky, arr[i]);
    }

    return 0;
}
