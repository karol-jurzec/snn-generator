#include "../../include/models/model_base.h"
#include <stdio.h>

void model_base_update(void *self, float input_current) {
    ModelBase *base = (ModelBase *)self;
    base->v += input_current;
}
