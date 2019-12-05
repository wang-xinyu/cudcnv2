#include "cudcnv2.h"
#include <map>
#include <iostream>
#include <fstream>

class Weights
{
    public:
        float *values_cpu; //!< The weight values, in a contiguous array.
        float *values_gpu;
        int count;      //!< The number of weights in the array.
};

// Load weights from files.
// weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);

    // Read number of weight blobs
    int32_t count;
    input >> count;

    while (count--)
    {
        Weights wt{nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values_cpu = (float*)val;
        cudaMalloc(&wt.values_gpu, size * sizeof(float));
        cudaMemcpy(wt.values_gpu, wt.values_cpu, size * sizeof(float), cudaMemcpyHostToDevice);

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}


int main() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    std::map<std::string, Weights> weightMap = loadWeights("../dcnv2.wts");

    CuDcnv2 *dcn = new CuDcnv2(64, 64, 3, 3, 1, 1, 1, 1, 1, 1, 1, weightMap["dcnv2.weight"].values_cpu, weightMap["dcnv2.bias"].values_cpu);

    int input_dims[4] = {1, 64, 128, 128};
    float *output_gpu;
    int output_size;

    dcn->forward_gpu(stream, handle, weightMap["input"].values_gpu, input_dims, weightMap["offset"].values_gpu, weightMap["mask"].values_gpu, &output_gpu, &output_size);

    float *output_cpu = new float[output_size];
    cudaMemcpy(output_cpu, output_gpu, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "\nOutput:\n\n";
    for (int i = 0; i < output_size; i++) {
        std::cout << output_cpu[i] << ", ";
        if (i % 10 == 0) std::cout << i / 10 << std::endl;
    }
    std::cout << std::endl;

    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values_cpu));
        cudaFree(mem.second.values_gpu);
    }
    cudaFree(output_gpu);
    delete[] output_cpu;
    cublasDestroy(handle);
    cudaStreamDestroy(stream);

    return 0;
}
