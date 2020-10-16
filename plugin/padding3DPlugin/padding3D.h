#pragma once

#include "NvInfer.h"

using namespace nvinfer1;
//using namespace nvinfer1::cudnn;
//just use for padding3D/fp32
namespace cuPad
{
    // Integer division rounding up
    inline __host__ __device__ constexpr int divUp(int x, int n)
    {
            return (x + n - 1) / n;
    }

    void cuPadding3D(void* dst, nvinfer1::Dims dstStride, const void* src, nvinfer1::Dims srcStride,
        int batchSize, const Dims &outputDims,
        const Dims& prePadding, const Dims& postPadding, int lgScalarsPerElement, cudaStream_t stream);
} // namespace cuPad


