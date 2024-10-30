#include <cmath>
#include <cstddef>
#include <cstdio>

#include "fcl.cuh"
#include "util.cuh"

void run(void) {
    size_t dims[] = { 784, 1000, 1000, 1000, 10 };
    size_t num_dims = sizeof(dims) / sizeof(size_t);
    fcl_layer *layers;
    cudaMallocManaged(&layers, (num_dims - 1) * sizeof(fcl_layer));

    // the first dimension is the number of inputs for which we don't create a layer
    for (size_t dim_i = 0; dim_i < num_dims - 1; dim_i++) {
        size_t dim = dims[dim_i + 1];
        size_t dim_prev = dims[dim_i];

        fcl_initialize(&layers[dim_i], dim, dim_prev);
        printf("Created layer %lu with dimensions of %lux%lu\n", dim_i, dim, dim_prev);
    }

    printf("Created network with %lu layers\n", num_dims - 1);

    /*
    for (size_t row = 0; row < layer.rows; row++) {
        for (size_t col = 0; col < layer.cols; col++) {
            printf("%10.6f ", layer.weights[row * layer.cols + col]);
        }

        printf("%10.6f\n", layer.bias[row]);
    }
    */

    float *activations;
    size_t dims_sum = 0;
    for (size_t dim_i = 0; dim_i < num_dims; dim_i++) {
        dims_sum += dims[dim_i];
    }

    cudaMallocManaged(&activations, dims_sum * sizeof(float));

    for (size_t input_i = 0; input_i < dims[0]; input_i++) {
        activations[input_i] = sample_gaussian(-0.3f, 2.5f);
    }

    printf("Allocated activations and filled input with random garbage\n");

    int threads_per_block = 256;

    size_t offset = 0;
    for (size_t layeri = 0; layeri < num_dims - 1; layeri++) {
        const fcl_layer *layer = &layers[layeri];
        int blocks = (layer->cols + threads_per_block - 1) / threads_per_block;
        printf("%lu blocks\n", layer->rows);

        fcl_feedforward<<<blocks, threads_per_block>>>(layer, &activations[offset], &activations[offset + layer->cols]);
        cudaDeviceSynchronize();

        offset += layer->cols;
        printf("Feedforwarded layer %lu\n", layeri);
    }

    for (size_t output_i = 0; output_i < dims[num_dims - 1]; output_i++) {
        printf("%10.6f\n", activations[output_i + offset]);
    }
}

int main (int argc, char *argv[]) {
    run();
}
