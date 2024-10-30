#include <cmath>
#include <cstdlib>

float sample_gaussian(float mean, float std) {
    // Box-Muller transform
    float u = rand() / (float)RAND_MAX;
    float v = rand() / (float)RAND_MAX;
    return sqrt(-2.0f * log(u)) * cos(2.0f * M_PI * v) * std + mean;
}
