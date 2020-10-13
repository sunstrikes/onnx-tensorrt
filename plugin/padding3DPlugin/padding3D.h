#pragma once

#include "NvInfer.h"

using namespace nvinfer1;
//using namespace nvinfer1::cudnn;
//just use for padding3D/fp32
namespace cuPad
{
    void cuPadding3D(void* dst, nvinfer1::Dims dstStride, const void* src, nvinfer1::Dims srcStride,
        int batchSize, const Dims &outputDims,
        const Dims3& prePadding, const Dims3& postPadding, int lgScalarsPerElement, cudaStream_t stream);
} // namespace cuPad

