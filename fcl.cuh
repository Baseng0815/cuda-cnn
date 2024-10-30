#pragma once

// Fully-connected layer

struct fcl_layer {
    float *weights;
    float *bias;
    size_t rows; // number of neurons in this layer
    size_t cols; // number of inputs (i.e. neurons in previous layer)

    float (*activation)(float weighted_sum);
};

__global__
void fcl_feedforward(const fcl_layer *layer, const float *act_in, float *act_out);

__global__
void fcl_backprop(const fcl_layer *layer, const float *weights_next, const float *error_next, const float *weighted_sum);

void fcl_initialize(fcl_layer *layer, size_t rows, size_t cols);
