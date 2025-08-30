#include "../layers/conv2d_layer.h"
#include <stdlib.h>
#include <stdio.h>

void apply_channel_compression(Conv2DLayer *layer, bool *inactive_out, bool *inactive_in);
