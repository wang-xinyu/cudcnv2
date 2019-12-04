#ifndef DCN_V2_CUDA
#define DCN_V2_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef __cplusplus
extern "C"
{
#endif
void dcn_v2_cuda_forward(cudaStream_t stream, cublasHandle_t handle,
        float *input, int input_dims[4],
        float *weight,
        float *bias, float *ones,
        float *offset, float *mask,
        float *output, float *columns,
        int cho,
        int kernel_h, int kernel_w,
        const int stride_h, const int stride_w,
        const int pad_h, const int pad_w,
        const int dilation_h, const int dilation_w,
        const int deformable_group);
#ifdef __cplusplus
}
#endif

#endif
