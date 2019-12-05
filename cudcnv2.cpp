#include "cudcnv2.h"
#include "cuda/dcn_v2_im2col_cuda.h"

// author: Wang Xinyu
// https://github.com/wang-xinyu/cudcnv2

void adjustLd(char transa, char transb, int m, int n, int k, int *lda, int *ldb, int *ldc) {
    int transa_ = ((transa == 't') || (transa == 'T'));
    int transb_ = ((transb == 't') || (transb == 'T'));

    if(n == 1)
        *ldc = m;

    if(transa_)
    {
        if(m == 1)
            *lda = k;
    }
    else
    {
        if(k == 1)
            *lda = m;
    }

    if(transb_)
    {
        if(k == 1)
            *ldb = n;
    }
    else
    {
        if(n == 1)
            *ldb = k;
    }
}

void dcn_v2_cuda_forward(cudaStream_t stream, cublasHandle_t handle,
                         const float *input, const int input_dims[4],
                         const float *weight, const float *bias, const float *ones,
                         const float *offset, const float *mask,
                         float *output, float *columns,
                         const int out_channels, const int height_out, const int width_out,
                         const int kernel_h, const int kernel_w,
                         const int stride_h, const int stride_w,
                         const int pad_h, const int pad_w,
                         const int dilation_h, const int dilation_w,
                         const int deformable_group)
{
    // ignore some dimension checks, you must ensure the dimensions input, weight, bias, ... are right.
    const int batch = input_dims[0];
    const int channels = input_dims[1];
    const int height = input_dims[2];
    const int width = input_dims[3];

    const float *input_n = input;
    const float *offset_n = offset;
    const float *mask_n = mask;
    float *output_n = output;

    for (int b = 0; b < batch; b++)
    {
        // Do Bias first:
        // M,N,K are dims of matrix A and B
        // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
        // (N x 1) (1 x M)
        int m_ = out_channels;
        int n_ = height_out * width_out;
        int k_ = 1;
        float alpha = 1.0;
        float beta = 0.0;
        int lda = k_;
        int ldb = k_;
        int ldc = n_;
        adjustLd('t', 'n', n_, m_, k_, &lda, &ldb, &ldc);

        cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n_, m_, k_,
                &alpha,
                ones, lda,
                bias, ldb,
                &beta,
                output_n, ldc);

        modulated_deformable_im2col_cuda(stream,
                input_n, offset_n,
                mask_n,
                1, channels, height, width,
                height_out, width_out, kernel_h, kernel_w,
                pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                deformable_group, columns);

        //(k * m)  x  (m * n)
        // Y = WC
        int m = out_channels;
        int n = height_out * width_out;
        int k = channels * kernel_h * kernel_w;
        alpha = 1.0;
        beta = 1.0;
        lda = n;
        ldb = k;
        ldc = n;
        adjustLd('n', 'n', n, m, k, &lda, &ldb, &ldc);
        cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                columns, lda,
                weight, ldb,
                &beta,
                output_n, ldc);

        input_n += channels * height * width;
        offset_n += height_out * width_out * deformable_group * kernel_h * kernel_w * 2;
        mask_n += height_out * width_out * deformable_group * kernel_h * kernel_w;
        output_n += m*n;
    }
}

CuDcnv2::CuDcnv2(int in_channels, int out_channels, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int deformable_groups, float *weights_cpu, float *bias_cpu) {
    in_channels_ = in_channels;
    out_channels_ = out_channels;
    kernel_h_ = kernel_h;
    kernel_w_ = kernel_w;
    stride_h_ = stride_h;
    stride_w_ = stride_w;
    pad_h_ = pad_h;
    pad_w_ = pad_w;
    dilation_h_ = dilation_h;
    dilation_w_ = dilation_w;
    deformable_groups_ = deformable_groups;

    cudaMalloc(&weights_gpu_, in_channels_ * out_channels_ * kernel_h_ * kernel_w_ * sizeof(float));
    cudaMalloc(&bias_gpu_, out_channels_ * sizeof(float));
    cudaMemcpy(weights_gpu_, weights_cpu, in_channels_ * out_channels_ * kernel_h_ * kernel_w_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_gpu_, bias_cpu, out_channels_ * sizeof(float), cudaMemcpyHostToDevice);
}

CuDcnv2::~CuDcnv2() {
    cudaFree(weights_gpu_);
    cudaFree(bias_gpu_);
}

void CuDcnv2::forward_gpu(cudaStream_t stream, cublasHandle_t handle, float *input_gpu, int input_dims[4], float *offset_gpu, float *mask_gpu, float **output_gpu, int *output_size) {
    int height_out = (input_dims[2] + 2 * pad_h_ - (dilation_h_ * (kernel_h_ - 1) + 1)) / stride_h_ + 1;
    int width_out = (input_dims[3] + 2 * pad_w_ - (dilation_w_ * (kernel_w_ - 1) + 1)) / stride_w_ + 1;

    float *ones_cpu = new float[height_out * width_out];
    for (int i = 0; i < height_out * width_out; i++) {
        ones_cpu[i] = 1.0;
    }
    float *ones_gpu;
    cudaMalloc(&ones_gpu, height_out * width_out * sizeof(float));
    cudaMemcpy(ones_gpu, ones_cpu, height_out * width_out * sizeof(float), cudaMemcpyHostToDevice);

    float *columns_gpu;
    cudaMalloc(&columns_gpu, input_dims[1] * kernel_h_ * kernel_w_ * height_out * width_out * sizeof(float));
    float *_output_gpu;
    cudaMalloc(&_output_gpu, input_dims[0] * out_channels_ * height_out * width_out * sizeof(float));

    dcn_v2_cuda_forward(stream, handle,
            input_gpu, input_dims,
            weights_gpu_, bias_gpu_, ones_gpu,
            offset_gpu, mask_gpu,
            _output_gpu, columns_gpu,
            out_channels_, height_out, width_out,
            kernel_h_, kernel_w_,
            stride_h_, stride_w_,
            pad_h_, pad_w_,
            dilation_h_, dilation_w_,
            deformable_groups_);
    *output_gpu = _output_gpu;
    *output_size = input_dims[0] * out_channels_ * height_out * width_out;

    cudaFree(columns_gpu);
    cudaFree(ones_gpu);
    delete[] ones_cpu;
}
