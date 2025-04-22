#include "../network.h"
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>


void log_weights(Network *network, int epoch, int batch);
void log_gradients(Network *network, int epoch, int sample);
void log_inputs(Network *network, int epoch, int sample, int t);
void log_outputs(Network *network, int epoch, int sample, int t);
void log_spikes(Network *network, int epoch, int sample, int t); 
void log_membranes(Network *network, int epoch, int sample, int t);
