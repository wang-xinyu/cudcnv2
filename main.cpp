#include "dcn_v2_cuda.h"

class Weights
{
    public:
        const float* cpu_values; //!< The weight values, in a contiguous array.
        void *gpu_values
        int64_t count;      //!< The number of weights in the array.
};

// Load weights from files.
// weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(onst std::string file)
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
        wt.cpu_values = val;
        cudaMalloc(&wt.gpu_values, size * sizeof(float));
        cudaMemcpy(wt.gpu_values, wt.cpu_values, size * sizeof(float), cudaMemcpyHostToDevice);

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

    std::map<std::string, Weights> weightMap = loadWeights("../dcnv2.wts");

    int input_dims[4] = {1, 64, 128, 128};

    int kernel_h = 3;
    int kernel_w = 3;
    int pad_h = 1;
    int pad_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    int deformable_group = 1;
    int channels_out = 64;

    int height_out = (input_dims[2] + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_out = (input_dims[3] + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    float *cpu_ones = new float[height_out * width_out];
    void *ones;
    cudaMalloc(&ones, height_out * width_out * sizeof(float));
    cudaMemcpy(ones, cpu_ones, height_out * width_out * sizeof(float), cudaMemcpyHostToDevice);

    void *columns;
    cudaMalloc(&columns * sizeof(float));

    dcn_v2_cuda_forward(stream, handle,
            weightMap["input"], input_dims,
            weightMap["dcnv2.weight"],
            weightMap["dcnv2.bias"], (float *)ones,
            weightMap["offset"], weightMap["mask"],
            float *output, float *columns,
            channels_out,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dilation_h, dilation_w,
            deformable_group);

    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.cpu_values));
        cudaFree(mem.second.gpu_values);
    }


    return 0;
}
