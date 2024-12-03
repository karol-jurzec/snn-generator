#include <stdio.h> 

typedef struct ModelBase {
    float v;           
    void (*update_neuron)(void *self, float input_current);
} ModelBase;