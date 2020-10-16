#include "padding3D.h"
#include "reducedMath.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <cassert>
#include <iostream>

using nvinfer1::rt::reduced_divisor;

namespace cuPad
{

namespace {

//template <int Unroll, typename T>
//__device__ __forceinline__ void applyScale(T tmp[Unroll], int c, const float* inI8Scale, const float* outI8Scale)
//{
//    assert(sizeof(T) > 1);
//}
//
//template <int Unroll>
//__device__ __forceinline__ void applyScale(int8_t tmp[Unroll], int c, const float* inI8Scale, const float* outI8Scale)
//{
//    assert(inI8Scale != nullptr);
//    assert(outI8Scale != nullptr);
//    // Dealing with Int8 data -- scale elements of the chunk
//#pragma unroll Unroll
//    for (int i = 0; i < Unroll; ++i)
//    {
//        float x = toFloat(tmp[i]);
//        int offset = c * Unroll + i;
//        x *= (inI8Scale[offset] / outI8Scale[offset]);
//        tmp[i] = fromFloat<int8_t>(x);
//    }
//}



struct float32
{
    float x[32];
};

template <typename U>
__device__ __forceinline__ U defaultValue(bool negativeInfinityPadding)
{
        return U();
}

template <>
__device__ __forceinline__ float defaultValue(bool negativeInfinityPadding)
{
    return 0.0f;
}

template <>
__device__ __forceinline__ float32 defaultValue<float32>(bool negativeInfinityPadding)
{
    float32 zero;
    #pragma unroll
    for (size_t i = 0; i < 32; i++)
    {
        zero.x[i] = 0.0f;
    }
    return zero;
}

} // namespace

/** Padding kernel.

      \param T actual type of the data
      \param U integer type that can hold a small vector of type T.  This is the type
      used to load/store data from/into the imag
      \param BlockDim number of threads per block
      \param isCHW true if data is in CHW format, false if data is in HWC format.

      \param dst pointer to output image.
      \param dstStride stride of output tensor, dstStride.d[0] is dstNStride, it is in units of U.
      \param src pointer to input image.
      \param srcStride stride of input tensor, srcStride.d[0] is srcNStride, it is in units of U.
      \param N number of images
      \param C dimension of input, in units of U.  Same as C dimension of output.
      For example, if T is int8_t and U is int32_t, and there are 100 int8_t
      channels, then C is 25.
      \param P height of output image
      \param Q width of output image
      \param preH amount of prepadding in height
      \param preW amount of prepadding in width
      \param postH amount of postpadding in height
      \param postW amount of postpadding in width
      \param divCPQ for div/mod by C*P*Q
      \param divPQ_or_C for div/mod by P*Q when isCHW=true; by C when isCHW=false
      \param divQ for div/mod by Q
      */
template <typename T, typename U, int BlockDim, bool isCHW>
__global__ __launch_bounds__(BlockDim) void pad(U* dst, nvinfer1::Dims dstStride,
                                                const U* src, nvinfer1::Dims srcStride,
                                                int N, int F, int C, int P, int Q, // output dimensions
                                                int preC, int preH, int preW, int postC, int postH, int postW,
                                                reduced_divisor divFCPQ, reduced_divisor divCPQ, reduced_divisor divPQ, 
                                                reduced_divisor divQ, bool negativeInfinityPadding = false)
{
    int tid = blockIdx.x * BlockDim + threadIdx.x;

    int n, f, c, fcpq, cpq, pq;
    //isCHW always true
    divFCPQ.divmod(tid, n, fcpq);
    divCPQ.divmod(fcpq, f, cpq);
    divPQ.divmod(cpq, c, pq);

    // Set (C,H,W) to input image size.
    int rH = P - postH - preH;
    int rW = Q - postW - preW;
    int rC = C - postC - preC;

    if (n < N)
    {
        int p, q;
        divQ.divmod(pq, p, q);

        int h = p - preH;
        int w = q - preW;
        int indc = c - preC;
        int dstIndex = n * dstStride.d[0] + f * dstStride.d[1] + c * dstStride.d[2] + p * dstStride.d[3] + q * dstStride.d[4];
        int srcIndex = n * srcStride.d[0] + f * srcStride.d[1] + indc * srcStride.d[2] + h * srcStride.d[3] + w * srcStride.d[4];
        const int Unroll = sizeof(U) / sizeof(T);
        T tmp[Unroll];

        // Load a chunk
        *reinterpret_cast<U*>(tmp) = indc >= 0 && indc < rC && h >= 0 && h < rH && w >= 0 && w < rW? src[srcIndex] : defaultValue<U>(negativeInfinityPadding);
        // Scale the chunk if necessary
        //applyScale<Unroll>(tmp, c, inI8Scale, outI8Scale);

        // Store the chunk
        dst[dstIndex] = *reinterpret_cast<U*>(tmp);
    }
}

// T is the scalar type of the data.
// U is the chunk type (either T or an integral type that holds 1<<LgPerVector items of type T).
template <typename T, typename U, int LgPerVector, bool isCHW = true>
void sublaunch(void* dst, nvinfer1::Dims dstStride, const void* src, nvinfer1::Dims srcStride,
               int batchSize, const Dims& outputDims,
               const Dims& prePadding, const Dims& postPadding,
               const cudaStream_t stream) {
    assert(sizeof(U) == sizeof(T) << LgPerVector);
    assert((dstStride.d[0] & ((1 << LgPerVector) - 1)) == 0);
    assert((srcStride.d[0] & ((1 << LgPerVector) - 1)) == 0);
    assert(reinterpret_cast<uintptr_t>(src) % sizeof(U) == 0);
    assert(reinterpret_cast<uintptr_t>(dst) % sizeof(U) == 0);

    const int block = 128;

    int N = batchSize;
    int F = outputDims.d[1]; //Frame
    int C = outputDims.d[2];
    int P = outputDims.d[3];
    int Q = outputDims.d[4];

    // Convert C and strides from being in units of T to units of U.
    F = (F + ((1 << LgPerVector) - 1)) >> LgPerVector;
    C = (C + ((1 << LgPerVector) - 1)) >> LgPerVector;
    dstStride.d[0] >>= LgPerVector;
    srcStride.d[0] >>= LgPerVector;

    reduced_divisor divFCPQ(F * C * P * Q);
    reduced_divisor divCPQ(C * P * Q);
    reduced_divisor divPQ(P * Q);
    reduced_divisor divQ(Q);
    size_t size = N * F * C * P * Q;

    pad<T, U, block, isCHW><<<divUp(size, block), block, 0, stream>>>((U*) dst, dstStride, (U*) src, srcStride,
                                                                      N, F, C, P, Q, prePadding.d[0], prePadding.d[1], prePadding.d[2], 
                                                                      postPadding.d[0], postPadding.d[1], postPadding.d[2], divFCPQ, divCPQ,
                                                                      divPQ, divQ);
}

/** Asymmetric padding interface between C++ and CUDA.

    \param dst pointer to output images
    \param dstStride stride of output tensor.
    \param src pointer to input images
    \param srcStride stride of input tensor
    \param batchSize number of input images
    \param outputDims dimensions of output
    \param prePadding amount of padding before image (negative values denote trimming)
    \param postPadding amount of padding after image (negative values denote trimming)
    \param lgScalarsPerElement log-2 of number of scalars per vector element.
    If the value is 3, the format is assumed to be NHWC where C must be a multiple of 8.
    Otherwise the format is assumed to be NCHW.
    \param stream CUDA stream
    */
void cuPadding3D(void* dst, nvinfer1::Dims dstStride, const void* src, nvinfer1::Dims srcStride,
            int batchSize, const Dims& outputDims,
            const Dims& prePadding, const Dims& postPadding, 
            int lgScalarsPerElement, const cudaStream_t stream) {
    if (lgScalarsPerElement == 0) {
        sublaunch<float, float, 0>(dst, dstStride, src, srcStride, batchSize, outputDims, prePadding, postPadding, stream);
    } else {
        std::cout << "Unsupported format for padding layer." << std::endl;
    }
}

} // namespace cuPad

