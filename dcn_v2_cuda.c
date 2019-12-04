#include "dcn_v2_cuda.h"
#include "cuda/dcn_v2_im2col_cuda.h"

// author: Wang Xinyu
// https://github.com/wang-xinyu/

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
                         const int deformable_group)
{
    const int batch = input_dims[0];
    const int channels = input_dims[1];
    const int height = input_dims[2];
    const int width = input_dims[3];
    const int channels_out = cho;
    // ignore some dimension checks, you should ensure the dimensions input, weight, bias, ... are right.

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    //if (THCudaTensor_nDimension(state, ones) != 2 ||
    //    THCudaTensor_size(state, ones, 0) * THCudaTensor_size(state, ones, 1) < height_out * width_out)
    //{
        // Resize plane and fill with ones...
    //    THCudaTensor_resize2d(state, ones, height_out, width_out);
    //    THCudaTensor_fill(state, ones, 1);
    //}

    // resize output
    //THCudaTensor_resize4d(state, output, batch, channels_out, height_out, width_out);
    // resize temporary columns
    //THCudaTensor_resize2d(state, columns, channels * kernel_h * kernel_w, 1 * height_out * width_out);

    // -- the memory of output and columns should be alloc before

    //THCudaTensor *input_n = THCudaTensor_new(state);
    //THCudaTensor *offset_n = THCudaTensor_new(state);
    //THCudaTensor *mask_n = THCudaTensor_new(state);
    //THCudaTensor *output_n = THCudaTensor_new(state);
    float *input_n = input;
    float *offset_n = offset;
    float *mask_n = mask;
    float *output_n = output;

    for (int b = 0; b < batch; b++)
    {
        //THCudaTensor_select(state, input_n, input, 0, b);
        //THCudaTensor_select(state, offset_n, offset, 0, b);
        //THCudaTensor_select(state, mask_n, mask, 0, b);
        //THCudaTensor_select(state, output_n, output, 0, b);

        // Do Bias first:
        // M,N,K are dims of matrix A and B
        // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
        // (N x 1) (1 x M)
        int m_ = channels_out;
        int n_ = height_out * width_out;
        int k_ = 1;
        float alpha = 1.0;
        float beta = 0.0;
        //THCudaBlas_Sgemm(state, 't', 'n', n_, m_, k_, 1.0f,
        //        THCudaTensor_data(state, ones), k_,
        //        THCudaTensor_data(state, bias), k_, 0.0f,
        //        THCudaTensor_data(state, output_n), n_);

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
        int m = channels_out;
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
        //THCudaBlas_Sgemm(state, 'n', 'n', n, m, k, 1.0f,
        //        THCudaTensor_data(state, columns), n,
        //        THCudaTensor_data(state, weight), k, 1.0f,
        //        THCudaTensor_data(state, output_n), n);
        input_n += channels * height * width;
        offset_n += height_out * width_out * deformable_group * kernel_h * kernel_w * 2;
        mask_n += height_out * width_out * deformable_group * kernel_h * kernel_w;
        output_n += ldc;
    }
}

