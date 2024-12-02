#include "model_base.c"

typedef struct {
    float u;         // Recovery variable
    float a, b, c, d; // Model parameters
    int spiked;      // Spiking flag

    ModelBase model_base; 
} IzhikevichNeuron;

void initialize_izhikevich(IzhikevichNeuron *neuron, float a, float b, float c, float d, float v_rest) {
    neuron->a = a;
    neuron->b = b;
    neuron->c = c;
    neuron->d = d;
    neuron->u = b * v_rest;
    neuron->spiked = 0;
    neuron->model_base.v = v_rest;
    neuron->model_base.update_neuron = update_izhikevich;
}

void update_izhikevich(void *self, float input_current) {
    IzhikevichNeuron* neuron = (IzhikevichNeuron*)self; 

    if (neuron->model_base.v >= 30.0) { // Spike threshold
        neuron->model_base.v = neuron->c;
        neuron->u += neuron->d;
        neuron->spiked = 1;
    } else {
        float dv = 0.04 * neuron->model_base.v * neuron->model_base.v + 5 * neuron->model_base.v + 140 - neuron->u + input_current;
        float du = neuron->a * (neuron->b * neuron->model_base.v - neuron->u);
        neuron->model_base.v += dv;
        neuron->u += du;
        neuron->spiked = 0;
    }
}
