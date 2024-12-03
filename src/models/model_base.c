#include "../../include/models/model_base.h"
#include <stdio.h>

// Example generic update function (not typically called directly)
void model_base_update(void *self, float input_current) {
    ModelBase *base = (ModelBase *)self;
    base->v += input_current;
    printf("Base neuron updated: V = %f\n", base->v);
}
