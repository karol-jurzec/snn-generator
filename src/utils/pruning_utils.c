#include "../../include/utils/pruning_utils.h"


void apply_channel_compression(Conv2DLayer *layer, bool *inactive_out, bool *inactive_in) {
    // build output channel mapping
    int active_out_count = 0;
    for (int i = 0; i < layer->original_out_channels; i++) {
        if (!inactive_out || !inactive_out[i]) {
            active_out_count++;
        }
    }
    
    if (active_out_count < layer->original_out_channels) {
        layer->out_active_channels_idx = (int*)malloc(active_out_count * sizeof(int));
        int idx = 0;
        for (int i = 0; i < layer->original_out_channels; i++) {
            if (!inactive_out || !inactive_out[i]) {
                layer->out_active_channels_idx[idx++] = i;
            }
        }
        layer->out_channels = active_out_count; 
    }
    
    //build input channel mapping  
    int active_in_count = 0;
    for (int i = 0; i < layer->original_in_channels; i++) {
        if (!inactive_in || !inactive_in[i]) {
            active_in_count++;
        }
    }
    
    if (active_in_count < layer->original_in_channels) {
        layer->in_active_channels_idx = (int*)malloc(active_in_count * sizeof(int));
        int idx = 0;
        for (int i = 0; i < layer->original_in_channels; i++) {
            if (!inactive_in || !inactive_in[i]) {
                layer->in_active_channels_idx[idx++] = i;
            }
        }
        layer->in_channels = active_in_count;   
    }
    
    printf("compressed Conv2D: %d→%d out_channels, %d→%d in_channels\n",
           layer->original_out_channels, layer->out_channels,
           layer->original_in_channels, layer->in_channels);
}
