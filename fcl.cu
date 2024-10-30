#include "fcl.cuh"
#include "util.cuh"

#include <stdio.h>

typedef float (*activation_function)(float);

__device__
static float relu(float weighted_sum);

__device__
static activation_function relu_device_ptr = relu;

__global__
void fcl_feedforward(const fcl_layer *layer, const float *act_in, float *act_out) {
    size_t row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= layer->rows) {
        return;
    }

    float weighted_sum = layer->bias[row];
    for (size_t col = 0; col < layer->cols; col++) {
        weighted_sum += layer->weights[row * layer->cols + col];
    }

    float activation = layer->activation(weighted_sum);
    act_out[row] = activation;
}

__global__
void fcl_backprop(const fcl_layer *layer, const float *weights_next, const float *error_next, const float *weighted_sum)
{
    size_t row = threadIdx.x + blockIdx.x * blockDim.x;

}

void fcl_initialize(fcl_layer *layer, size_t rows, size_t cols) {
    layer->rows = rows;
    layer->cols = cols;

    cudaMallocManaged(&layer->weights, layer->rows * layer->cols * sizeof(float));
    cudaMallocManaged(&layer->bias, layer->rows * sizeof(float));

    float mean = 0.0f;
    float std = sqrt(2.0f / (float)rows);

    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
            // Kaiming initialization (Gaussian with std of sqrt(2/n))
            layer->weights[row * layer->cols + col] = sample_gaussian(mean, std);
        }

        layer->bias[row] = sample_gaussian(mean, std);
    }

    activation_function activation_host_ptr;
    cudaMemcpyFromSymbol(&activation_host_ptr, relu_device_ptr, sizeof(activation_host_ptr));
    layer->activation = activation_host_ptr;
}

__device__
float relu(float weighted_sum) {
    return fmax(0.0f, weighted_sum);
}
