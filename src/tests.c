#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

#include "../include/tests.h"
#include "../include/network.h"

#include "../include/models/model_base.h"
#include "../include/models/lif_neuron.h"

#include "../include/utils/snn_plot.h"
#include "../include/utils/layer_utils.h"
#include "../include/utils/network_loader.h"
#include "../include/utils/network_logger.h"
#include "../include/utils/nmnist_loader.h"
#include "../include/utils/dataset_loader.h"
#include "../include/utils/perf.h"

#include "../include/layers/conv2d_layer.h"
#include "../include/layers/maxpool2d_layer.h"
#include "../include/layers/flatten_layer.h"
#include "../include/layers/linear_layer.h"
#include "../include/layers/spiking_layer.h"


// Print the output feature map for inspection
void print_output(float *output, size_t out_channels, size_t output_dim) {
    for (size_t oc = 0; oc < out_channels; oc++) {
        printf("Output Channel %zu:\n", oc);
        for (size_t y = 0; y < output_dim; y++) {
            for (size_t x = 0; x < output_dim; x++) {
                printf("%0.2f ", output[(oc * output_dim * output_dim) + (y * output_dim + x)]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void print_output_spikes(float *output, size_t num_neurons) {
    printf("Output Spikes:\n");
    for (size_t i = 0; i < num_neurons; i++) {
        printf("Neuron %lu: Spiked = %0.1f\n", i, output[i]);
    }
    printf("\n");
}

void generate_synthetic_input(float *input, size_t size) {
    for (size_t i = 0; i < size; i++) {
        input[i] = (float)(i % 255) / 255.0f; // Normalize pixel values to [0, 1]
    }
}

void single_neuron_test(ModelBase *model_base, const char* filename) {
    int dt = 200;
    FILE *log_file = fopen("out/single_neuron_output.txt", "w");
    if (log_file == NULL) {
        perror("Error opening log file");
        return;
    }

        double arr[dt];
    for (int i = 0; i < dt; ++i) {
        if (i < 100) {
            arr[i] = 0.0;
        } else if (i < 150) {
            arr[i] = 0.25;
        } else {
            arr[i] = 0.0;
        }
    }

    for (int i = 0; i < dt; ++i) {
        model_base->update_neuron(model_base, arr[i]);  // Polymorphic call
        fprintf(log_file, "%d %f\n", i, model_base->v); // Log time and potential
    }

    fclose(log_file);
    plot_single_neuron("out/single_neuron_output.txt", filename);
}

void conv2d_test() {
    // // Parameters
    // int in_channels = 1; // Single channel input (e.g., grayscale image)
    // int out_channels = 2; // Two output channels for test
    // int kernel_size = 3;  // 3x3 kernel
    // int stride = 1;       // Stride of 1
    // int padding = 1;      // Padding of 1 (to preserve input size)

    // size_t input_size = 14 * 14; // Assume 28x28 input image (MNIST-like)
    // float *input = (float *)malloc(input_size * sizeof(float));

    // for (size_t i = 0; i < input_size; i++) {
    //     //input[i] = (float)(i % 14 + i % 14) / (float)(14 + 14); // Simple pattern
    //     input[i] = 1.0f;
    // }

    // // Initialize Conv2D layer
    // Conv2DLayer conv_layer;
    // conv2d_initialize(&conv_layer, in_channels, out_channels, kernel_size, stride, padding, 28);

    // // Perform forward pass
    // conv2d_forward(&conv_layer, input, input_size);

    // // Calculate output dimensions
    // size_t output_dim = calculate_output_dim(14, kernel_size, stride, padding);

    // // Print output feature map
    // print_output(conv_layer.output, out_channels, output_dim);

    // // Free resources
    // free(input);
    // conv2d_free(&conv_layer);
}

void maxpool2d_test() {
//     int kernel_size = 2;
//     int stride = 2;
//     int padding = 0;

//     size_t input_dim = 14;
//     size_t input_size = input_dim * input_dim * 1; // 1 channel input
//     float *input = (float *)malloc(input_size * sizeof(float));

//     for (size_t i = 0; i < input_size; i++) {
//         //input[i] = (float)(i % 14 + i % 14) / (float)(14 + 14);
//         input[i] = 1.0f;
//     }

//     MaxPool2DLayer pool_layer;
//     maxpool2d_initialize(&pool_layer, kernel_size, stride, padding, 28, 2);
//     maxpool2d_forward(&pool_layer, input, input_size); 

//     size_t output_dim = calculate_output_dim(input_dim, kernel_size, stride, padding);
//     print_output(pool_layer.output, 1, output_dim);

//     free(input);
//     maxpool2d_free(&pool_layer);
}

void flatten_test() {
    // size_t input_size = 4 * 4 * 2; // Example: 4x4 image with 2 channels (32 elements)
    // float *input = (float *)malloc(input_size * sizeof(float));

    // for (size_t i = 0; i < input_size; i++) {
    //     input[i] = (float)i / input_size; // Simple pattern
    // }

    // FlattenLayer flatten_layer;
    // flatten_initialize(&flatten_layer, input_size);

    // flatten_forward(&flatten_layer, input, input_size);

    // for (size_t i = 0; i < flatten_layer.output_size; i++) {
    //     printf("%0.2f ", flatten_layer.output[i]);
    // }
    // printf("\n");

    // free(input);
    // flatten_free(&flatten_layer);
}

void linear_test() {
    // size_t in_features = 16;  // Example input size
    // size_t out_features = 4;  // Example output size
    // float *input = (float *)malloc(in_features * sizeof(float));

    // for (size_t i = 0; i < in_features; i++) {
    //     input[i] = (float)i / in_features;
    // }

    // LinearLayer linear_layer;
    // linear_initialize(&linear_layer, in_features, out_features);

    // linear_forward(&linear_layer, input, in_features);

    // for (size_t i = 0; i < out_features; i++) {
    //     printf("%0.2f ", linear_layer.output[i]);
    // }
    // printf("\n");

    // free(input);
    // linear_free(&linear_layer);
}

void spiking_layer_test() {
    // size_t num_neurons = 5;
    // float *input = (float *)malloc(num_neurons * sizeof(float));
    // for (size_t i = 0; i < num_neurons; i++) {
    //     input[i] = (float)i / num_neurons;
    // }

    // // Create neuron models (using LeakyLIF as an example)
    // ModelBase *neuron_models[num_neurons];
    // for (size_t i = 0; i < num_neurons; i++) {
    //     neuron_models[i] = (ModelBase *)malloc(sizeof(LIFNeuron));
    //     lif_initialize((LIFNeuron *)neuron_models[i], 0.0f, 1.0f, 0.0f, 0.5f);
    // }

    // // Initialize spiking layer
    // SpikingLayer spiking_layer;
    // spiking_initialize(&spiking_layer, num_neurons, neuron_models);

    // // Perform forward pass
    // spiking_forward(&spiking_layer, input, num_neurons);
    // spiking_forward(&spiking_layer, input, num_neurons);

    // // Print spike outputs
    // printf("Spike Outputs:\n");
    // for (size_t i = 0; i < num_neurons; i++) {
    //     printf("Neuron %lu: Spiked = %0.1f\n", i, spiking_layer.output_spikes[i]);
    // }

    // // Free resources
    // for (size_t i = 0; i < num_neurons; i++) {
    //     free(neuron_models[i]);
    // }
    // free(input);
    // spiking_free(&spiking_layer);
}

void network_test() { 
    size_t input_size = 28 * 28 * 2; // Example input size for 2-channel image
    float *input = (float *)malloc(input_size * sizeof(float));

    for (size_t i = 0; i < input_size; i++) {
        input[i] = (float)i / input_size;
    }

    // Create a network with 3 layers
    Network *network = create_network(3);

    // Initialize and add Conv2D layer
    Conv2DLayer *conv_layer = (Conv2DLayer *)malloc(sizeof(Conv2DLayer));
    conv2d_initialize(conv_layer, 2, 12, 5, 1, 0, 24);
    add_layer(network, (LayerBase *)conv_layer, 0);

    // Initialize and add MaxPool2D layer
    MaxPool2DLayer *pool_layer = (MaxPool2DLayer *)malloc(sizeof(MaxPool2DLayer));
    maxpool2d_initialize(pool_layer, 2, 2, 0, 28, 2);
    add_layer(network, (LayerBase *)pool_layer, 1);

    // Initialize and add Flatten layer
    FlattenLayer *flatten_layer = (FlattenLayer *)malloc(sizeof(FlattenLayer));
    flatten_initialize(flatten_layer, conv_layer->output_size);
    add_layer(network, (LayerBase *)flatten_layer, 2);

    // Perform forward pass
    //forward(network, input, input_size);

    // Free resources
    free(input);
    free_network(network);
}

void network_loader_test() { 
    // // Load the network from the config file
    // Network *network = initialize_network_from_file("example_model.json");
    // if (!network) {
    //     printf("Failed to initialize network.\n");
    // }

    // // Generate synthetic input (28x28 image with 2 channels)
    // size_t input_size = 28 * 28 * 2;
    // float *input = (float *)malloc(input_size * sizeof(float));
    // generate_synthetic_input(input, input_size);

    // // Perform forward pass
    // for(int i = 0; i < 10; ++i) {
    //     forward(network, input, input_size);
    // }

    // // Assume last layer is a spiking layer and print spikes
    // SpikingLayer *last_layer = (SpikingLayer *)network->layers[network->num_layers - 1];
    // print_output_spikes(last_layer->base.output, last_layer->num_neurons);

    // // Free resources
    // free(input);
    // free_network(network); 
}

void print_sample(const Sample *sample, size_t max_events_to_display) {
    printf("Label: %d\n", sample->label);
    printf("Number of events: %zu\n", sample->num_events);

    size_t num_events_to_display = sample->num_events < max_events_to_display
                                       ? sample->num_events
                                       : max_events_to_display;
    for (size_t i = 0; i < num_events_to_display; i++) {
        printf("Event %zu: X=%d, Y=%d, Polarity=%d, Timestamp=%u\n", i + 1,
               sample->events[i].x, sample->events[i].y,
               sample->events[i].polarity, sample->events[i].timestamp);
    }

    if (sample->num_events > max_events_to_display) {
        printf("... (%zu more events not displayed)\n",
               sample->num_events - max_events_to_display);
    }
}

void loader_test() {
    //const char *dataset_dir = "C:/Users/karol/Desktop/karol/agh/praca_snn/dataset/dvsgesture/DvsGesture";  
    //size_t max_samples_to_load = 100;

    //Dataset *ds = load_dataset(dataset_dir, FORMAT_DVSGESTURE, 20, false);
    //printf("Loaded %zu samples\n", ds->num_samples);
}

void nmnist_loader_test() {
    // Directory containing the NMNIST dataset
    // const char *dataset_dir = "/Users/karol/Desktop/karol/agh/praca-snn/N-MNIST/Train";   
    const char *dataset_dir = "C:/Users/karol/Documents/datasets/N-MNIST/Test";   


    // Maximum number of samples to load for testing
    size_t max_samples_to_load = 10000;

    // Enable stabilization (true) or disable it (false)
    bool stabilize = false;
    bool denoise = true;

    printf("Loading NMNIST dataset with stabilization=%s...\n",
           stabilize ? "ENABLED" : "DISABLED");

    // Load the NMNIST dataset
    Dataset *dataset =
        load_nmnist_dataset(dataset_dir, max_samples_to_load, stabilize, denoise);

    if (!dataset) {
        printf("Error: Failed to load NMNIST dataset.\n");
        return;
    }

    printf("Loaded %zu samples from NMNIST dataset.\n", dataset->num_samples);

    // Display information for each loaded sample
    for (size_t i = 0; i < 10; i++) {
        //printf("\nSample %zu:\n", i + 1);
        if(&dataset->samples[i].label != 0) {
            //visualize_sample_frames(&dataset->samples[i], "out/sample_01_frames", 16, 28, 28, 100000);
            break;
        }
        //print_sample(&dataset->samples[i], 10); // Display up to 10 events per sample
    }

    // Free the dataset
    //free_nmnist_dataset(dataset);

    printf("NMNIST dataset successfully tested and freed.\n");
    return;
}

void discretization_test() {
    // Load the NMNIST dataset
    //NMNISTDataset *dataset = load_nmnist_dataset("/Users/karol/Desktop/karol/agh/praca-snn/N-MNIST/Train", 10, true);

   // Sample sample = load_nmnist_sample(
   // "/Users/karol/Desktop/karol/agh/praca-snn/N-MNIST/Train/5/00333.bin",
   // 4, false, true);

   // int max_time = sample.events[sample.num_events - 1].timestamp;

    //NMNISTSample sample = load_nmnist_sample(file_path, digit, stabilize);
   // visualize_sample_frames(&sample, "out/sample_0_frames", 311, 34, 34, max_time);
   // plot_event_grid(&sample, 2, 3, 0);
    
    /*
    // Visualize the first sample as temporal frames
    if (dataset->num_samples > 0) {
        visualize_sample_frames(&dataset->samples[1], "out/sample_1_frames", 16, 34, 34, 100000);
        visualize_sample_frames(&dataset->samples[2], "out/sample_2_frames", 16, 34, 34, 100000);
        visualize_sample_frames(&dataset->samples[3], "out/sample_3_frames", 16, 34, 34, 100000);
    }

    // Free the dataset
    free_nmnist_dataset(dataset);
    */
}

static double now_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void stmnist_test() {
	const char *model_architecrure = "scnn_stmnist_architecture.json";
	const char *model_weights = "scnn_stmnist_weights_bs_64.json";
	const char *dataset_path_test = "C:/Users/karol/Desktop/karol/agh/praca_snn/dataset/STMNIST/data_submission"; 

	printf("Loading network from %s...\n", model_architecrure);
	perf_mark_start("network_init");
	Network *network = initialize_network_from_file(model_architecrure, 10, 10, 2);
	perf_mark_end("network_init");
	if (!network) {
		printf("Error: Failed to load network.\n");
		return;
	}

	perf_mark_start("weights_load");
	load_weights_from_json(network, model_weights);
	perf_mark_end("weights_load");
	printf("Weights were read succesfully...\n");

	printf("Loading test dataset from %s...\n", dataset_path_test);
	perf_mark_start("dataset_load");
	Dataset *dataset_test = load_dataset(dataset_path_test, FORMAT_STMNIST, 10, false, false);
	perf_mark_end("dataset_load");
	if (dataset_test) {
		perf_add_metric("dataset_samples", (double)dataset_test->num_samples);
	}

	printf("Testing the network accuracy...\n");

	double t0 = now_seconds();

	perf_mark_start("inference_total");
	test(network, dataset_test);
	perf_mark_end("inference_total");

	double t1 = now_seconds();
	printf("train_test() took %.6f seconds\n", t1 - t0);

	optimize_network(network);

	Dataset *dataset_test_v2 = load_dataset(dataset_path_test, FORMAT_STMNIST, 100, false, false);

	perf_mark_start("inference_total");
	test(network, dataset_test_v2);
	perf_mark_end("inference_total");

	free_network(network);

	printf("Test was completed successfully.\n");
}


void nmnist_test() {
    const char *network_config_path_train = "C:/Users/karol/Desktop/karol/agh/praca_snn/data/NMNIST/Train";   
    const char *network_config_path_test = "C:/Users/karol/Desktop/karol/agh/praca_snn/data/NMNIST/Test"; 
    const char *dataset_path = "snn_nmnist_architecture.json";

    // Load the network
    printf("Loading network from %s...\n", dataset_path);
    perf_mark_start("nmnist_network_init");
    Network *network = initialize_network_from_file(dataset_path, 34, 34, 2);
    perf_mark_end("nmnist_network_init");
    if (!network) {
        printf("Error: Failed to load network.\n");
        return;
    }

    perf_mark_start("nmnist_weights_load");
    load_weights_from_json(network, "snn_nmnist_weights_bs_32.json");
    perf_mark_end("nmnist_weights_load");

    printf("Loading test dataset from %s...\n", network_config_path_test);
    perf_mark_start("nmnist_dataset_load");
    Dataset *dataset_test = load_dataset(network_config_path_test, FORMAT_NMNIST, 250, false, true);
    perf_mark_end("nmnist_dataset_load");
    if (dataset_test) {
        perf_add_metric("nmnist_dataset_samples", (double)dataset_test->num_samples);
    }

    printf("Testing the network accuracy...\n");
    perf_mark_start("nmnist_inference_total");
    float acc = test(network, dataset_test);  // Call after training
    free_dataset(dataset_test);
    perf_mark_end("nmnist_inference_total");
    perf_add_metric("nmnist_accuracy_percent", (double)acc);

    // Clean up
    //_dataset(dataset_test);
    //free_nmnist_dataset(dataset_train);
    free_network(network);

    printf("Training test completed successfully.\n");
}

// Simple synthetic data generator
float* generate_batch(int batch_size, int features, int* labels) {
    float* data = malloc(batch_size * features * sizeof(float));
    for (int i = 0; i < batch_size; i++) {
        float sum = 0;
        for (int j = 0; j < features; j++) {
            data[i * features + j] = (float)rand() / RAND_MAX;
            sum += data[i * features + j];
        }
        labels[i] = (sum / features) > 0.5f ? 1 : 0;
    }
    return data;
}

// Layer creation functions implementation
Conv2DLayer* create_conv2d_layer(int height, int width, int in_channels, int out_channels, 
                               int kernel_size, int stride, int padding) {
    Conv2DLayer *layer = (Conv2DLayer*)malloc(sizeof(Conv2DLayer));
    conv2d_initialize(layer, in_channels, out_channels, kernel_size, stride, padding, height);
    return layer;
}

MaxPool2DLayer* create_maxpool_layer(int height, int width, int channels, 
                                   int pool_size, int stride) {
    MaxPool2DLayer *layer = (MaxPool2DLayer*)malloc(sizeof(MaxPool2DLayer));
    maxpool2d_initialize(layer, pool_size, stride, 0, height, channels);
    return layer;
}

FlattenLayer* create_flatten_layer(size_t input_size) {
    FlattenLayer *layer = (FlattenLayer*)malloc(sizeof(FlattenLayer));
    flatten_initialize(layer, input_size);
    return layer;
}

LinearLayer* create_linear_layer(size_t in_features, size_t out_features) {
    LinearLayer *layer = (LinearLayer*)malloc(sizeof(LinearLayer));
    linear_initialize(layer, in_features, out_features);
    return layer;
}

SpikingLayer* create_spiking_layer(size_t num_neurons) {
    SpikingLayer *layer = (SpikingLayer*)malloc(sizeof(SpikingLayer));
    // Using LIF neurons as default
    ModelBase **neurons = (ModelBase**)malloc(num_neurons * sizeof(ModelBase*));
    for (size_t i = 0; i < num_neurons; i++) {
        neurons[i] = (ModelBase*)malloc(sizeof(LIFNeuron));
        // Initialize LIF neuron here

        lif_initialize((LIFNeuron *)neurons[i], 0.0f, 0.5f, 0.0f, 0.5f);
    }
    spiking_initialize(layer, num_neurons, neurons);
    return layer;
}

// Iris dataset parameters
#define MAX_LINE_LENGTH 100
#define NUM_FEATURES 4
#define NUM_CLASSES 3
#define NUM_SAMPLES 150
#define TRAIN_SIZE 120  // 80% of 150
#define TEST_SIZE 30    // 20% of 150

typedef struct {
    float sepal_length;
    float sepal_width;
    float petal_length;
    float petal_width;
    int variety;  // 0=Setosa, 1=Versicolor, 2=Virginica
} IrisSample;

void parse_iris_line(char *line, IrisSample *sample) {
    char *token;
    int col = 0;
    
    // Remove newline if present
    line[strcspn(line, "\n")] = 0;
    
    token = strtok(line, ",");
    while (token != NULL && col < 5) {
        // Remove quotes if present
        if (token[0] == '"') {
            memmove(token, token+1, strlen(token));  // Remove opening quote
            token[strlen(token)-1] = '\0';           // Remove closing quote
        }
        
        switch(col) {
            case 0:  // sepal.length
                sample->sepal_length = atof(token);
                break;
            case 1:  // sepal.width
                sample->sepal_width = atof(token);
                break;
            case 2:  // petal.length
                sample->petal_length = atof(token);
                break;
            case 3:  // petal.width
                sample->petal_width = atof(token);
                break;
            case 4:  // variety
                if (strcmp(token, "\"Setosa\"") == 0 || strcmp(token, "Setosa") == 0)
                    sample->variety = 0;
                else if (strcmp(token, "\"Versicolor\"") == 0 || strcmp(token, "Versicolor") == 0)
                    sample->variety = 1;
                else if (strcmp(token, "\"Virginica\"") == 0 || strcmp(token, "Virginica") == 0)
                    sample->variety = 2;
                break;
        }
        
        token = strtok(NULL, ",");
        col++;
    }
}

// ReLU activation function
void relu_forward(float *input, size_t size) {
    for (size_t i = 0; i < size; i++) {
        input[i] = input[i] > 0 ? input[i] : 0;
 
 
    }
}

// Softmax function
void softmax(float *input, size_t size) {
    float max_val = input[0];
    for (size_t i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        input[i] = expf(input[i] - max_val);
        sum += input[i];
    }
    
    for (size_t i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

// Cross-entropy loss
float cross_entropy_loss(float* predictions, int target, size_t num_classes) {
    float loss = 0.0f;
    for (size_t i = 0; i < num_classes; i++) {
        if (i == target) {
            loss += -logf(predictions[i] + 1e-10f); // Small epsilon to avoid log(0)
        }
    }
    return loss;
}

// Calculate accuracy
float calculate_accuracy(int *predicted, int *true_labels, size_t size) {
    int correct = 0;
    for (size_t i = 0; i < size; i++) {
        if (predicted[i] == true_labels[i]) {
            correct++;
        }
    }
    return (float)correct / size;
}

// helper to shuffle the dataset
void shuffle_data(float X[][NUM_FEATURES], int y[], int size, int num_features) {
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        
        // Swap features
        float temp_features[num_features];
        memcpy(temp_features, X[i], num_features * sizeof(float));
        memcpy(X[i], X[j], num_features * sizeof(float));
        memcpy(X[j], temp_features, num_features * sizeof(float));
        
        // Swap labels
        int temp_label = y[i];
        y[i] = y[j];
        y[j] = temp_label;
    }
}

void scale_gradients_by_batch(Network *network, int batch_size) {
    for (int l = 0; l < network->num_layers; l++) {
        LayerBase *layer = network->layers[l];

        // Cast the layer based on its type
        if (layer->layer_type == LAYER_CONV2D) {
            Conv2DLayer *conv_layer = (Conv2DLayer*)layer;
            for (int i = 0; i < conv_layer->base.num_weights; i++) {
                conv_layer->base.weight_gradients[i] /= (float)batch_size;
            }
            for (int i = 0; i < conv_layer->out_channels; i++) {
                conv_layer->base.bias_gradients[i] /= (float)batch_size;
            }
        } else if (layer->layer_type == LAYER_LINEAR) {
            LinearLayer *linear_layer = (LinearLayer*)layer;
            for (int i = 0; i < linear_layer->base.num_weights; i++) {
                linear_layer->base.weight_gradients[i] /= (float)batch_size;
            }
            for (int i = 0; i < linear_layer->out_features; i++) {
                linear_layer->base.bias_gradients[i] /= (float)batch_size;
            }
        } 
    }
}

void save_hidden_gradients_to_file(const float* hidden_gradients, int batch_num, int sample_num, int size, const char* filename) {
    FILE* file = fopen(filename, "a"); // Append mode
    if (file == NULL) {
        perror("Failed to open file");
        return;
    }

    fprintf(file, "Batch %d, Sample %d:\n", batch_num, sample_num);
    for (int i = 0; i < size; i++) {
        fprintf(file, "%f ", hidden_gradients[i]);
    }
    fprintf(file, "\n\n");

    fclose(file);
}

void save_output_to_file(const float* output, int batch_num, int sample_num, const char* filename) {
    FILE* file = fopen(filename, "a"); // Append mode
    if (file == NULL) {
        perror("Failed to open file");
        return;
    }

    fprintf(file, "Batch %d, Sample %d:\n", batch_num, sample_num);
    for (int i = 0; i < NUM_CLASSES; i++) {
        fprintf(file, "%f ", output[i]);
    }
    fprintf(file, "\n\n");

    fclose(file);
}









