#ifndef DCN_V2_CUDA
#define DCN_V2_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>

class CuDcnv2 {
    public:
        CuDcnv2(int in_channels, int out_channels, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int deformable_groups, float *weights_cpu, float *bias_cpu);
        ~CuDcnv2();
        void forward_gpu(cudaStream_t stream, cublasHandle_t handle, float *input_gpu, int input_dims[4], float *offset_gpu, float *mask_gpu, float **output_gpu, int *output_size);
private:
    int in_channels_;
    int out_channels_;
    int kernel_h_;
    int kernel_w_;
    int stride_h_;
    int stride_w_;
    int pad_h_;
    int pad_w_;
    int dilation_h_;
    int dilation_w_;
    int deformable_groups_;
    float *weights_gpu_;
    float *bias_gpu_;

};

#endif
