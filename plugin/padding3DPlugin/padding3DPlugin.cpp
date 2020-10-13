/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "padding3DPlugin.h"
#include "plugin.h"
#include <algorithm>
#include <cuda_runtime_api.h>
#include <iostream>

#define DEBUG 0

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::Padding3D;
using nvinfer1::plugin::Padding3DPluginCreator;

namespace
{
const char* PADDING3D_PLUGIN_VERSION{"1"};
const char* PADDING3D_PLUGIN_NAME{"Padding3D_TRT"};
} // namespace

PluginFieldCollection Padding3DPluginCreator::mFC{};
std::vector<PluginField> Padding3DPluginCreator::mPluginAttributes;

Padding3DPluginCreator::Padding3DPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("scale", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* Padding3DPluginCreator::getPluginName() const
{
    return PADDING3D_PLUGIN_NAME;
}

const char* Padding3DPluginCreator::getPluginVersion() const
{
    return PADDING3D_PLUGIN_VERSION;
}

const PluginFieldCollection* Padding3DPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* Padding3DPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    Dims3 prePadding, postPadding;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "prePadding")) {
            assert(fields[i].type == PluginFieldType::kDIMS);
            prePadding = *(static_cast<const Dims3*>(fields[i].data));
        }
        if (!strcmp(attrName, "postPadding")) {
            assert(fields[i].type == PluginFieldType::kDIMS);
            postPadding = *(static_cast<const Dims3*>(fields[i].data));
        }
    }
    return new Padding3D(prePadding, postPadding);
}

IPluginV2Ext* Padding3DPluginCreator::deserializePlugin(const char* name, const void* data, size_t length)
{
    return new Padding3D(data, length);
}

Padding3D::Padding3D(const Dims3& prePadding, const Dims3& prePadding) {
    mPrePadding = prePadding;
    mPostPadding = postPadding;
}

int Padding3D::getNbOutputs() const
{
    return 1;
}

Dims Padding3D::getOutputDimensions(int index, const Dims* inputDims, int nbInputs)
{
    assert(nbInputs == 1);
    nvinfer1::Dims const& input = inputDims[0];
    assert(index == 0);
    nvinfer1::Dims output;
    output.nbDims = input.nbDims;
    //last 3dim
    auto pad_dim = output.nbDims - 3;
    for (int d = 0; d < input.nbDims; ++d) {
        output.d[d] = input.d[d]; 
        if (d >= pad_dim) {
            auto ind = 3 - output.nbDims + d;
            assert(ind >= 0);
            output.d[d] = output.d[d] + mPrePadding[ind] + mPostPadding[ind];
        }
    }
    return output;
}

int Padding3D::initialize()
{
    return 0;
}

void Padding3D::terminate() {}

void Padding3D::destroy()
{
    delete this;
}

size_t Padding3D::getWorkspaceSize(int) const
{
    return 0;
}

size_t Padding3D::getSerializationSize() const
{
    // dimensions: 3 * 2
    return sizeof(int) * 3 * 2;
}

void Padding3D::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mInputDims.d[0]);
    write(d, mInputDims.d[1]);
    write(d, mInputDims.d[2]);
    write(d, mInputDims.d[3]);
    write(d, mInputDims.d[4]);
    write(d, mPrePadding.d[0]);
    write(d, mPrePadding.d[1]);
    write(d, mPrePadding.d[2]);
    write(d, mPostPadding.d[0]);
    write(d, mPostPadding.d[1]);
    write(d, mPostPadding.d[2]);
    ASSERT(d == a + getSerializationSize());
}

Padding3D::Padding3D(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mInputDims = Dims();
    mInputDims.d[0] = read<int>(d);
    mInputDims.d[1] = read<int>(d);
    mInputDims.d[2] = read<int>(d);
    mInputDims.d[3] = read<int>(d);
    mInputDims.d[4] = read<int>(d);
    mPrePadding = Dims3();
    mPrePadding.d[0] = read<int>(d);
    mPrePadding.d[1] = read<int>(d);
    mPrePadding.d[2] = read<int>(d);
    mPostPadding = Dims3();
    mPostPadding.d[0] = read<int>(d);
    mPostPadding.d[1] = read<int>(d);
    mPostPadding.d[2] = read<int>(d);
    ASSERT(d == a + length);
}

const char* Padding3D::getPluginType() const
{
    return "Padding3D_TRT";
}

const char* Padding3D::getPluginVersion() const
{
    return "1";
}

IPluginV2Ext* Padding3D::clone() const
{
    auto plugin = new Padding3D(*this);
    plugin->setPluginNamespace(mNameSpace.c_str());
    return plugin;
}

void Padding3D::setPluginNamespace(const char* libNamespace)
{
    mNameSpace = libNamespace;
}

const char* Padding3D::getPluginNamespace() const
{
    return mNameSpace.c_str();
}

bool Padding3D::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

int Padding3D::enqueue(
    int batch_size, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{

    //int nchan = mOutputDims.d[0];
    //float scale = mScale;
    //int2 osize = {mOutputDims.d[2], mOutputDims.d[1]};
    //int istride = mInputDims.d[2];
    //int ostride = mOutputDims.d[2];
    //int ibatchstride = mInputDims.d[1] * istride;
    //int obatchstride = mOutputDims.d[1] * ostride;
    //dim3 block(32, 16);
    //dim3 grid((osize.x - 1) / block.x + 1, (osize.y - 1) / block.y + 1, std::min(batch_size * nchan, 65535));

    //resizeNearest(grid, block, stream, batch_size * nchan, scale, osize, static_cast<float const*>(inputs[0]), istride,
    //    ibatchstride, static_cast<float*>(outputs[0]), ostride, obatchstride);
    mOutputDims = getOutputDimensions(0, &mInputDims, 1);
    cuPadding3D(outputs[0], mOutputDims, inputs[0], batch_size, mInputDims, mPrePadding, mPostPadding, 0, stream);

    return cudaGetLastError() != cudaSuccess;
}

// Return the DataType of the plugin output at the requested index
DataType Padding3D::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Only 1 input and 1 output from the plugin layer
    ASSERT(index == 0);

    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool Padding3D::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool Padding3D::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Configure the layer with input and output data types.
void Padding3D::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    assert(nbInputs == 1);
    mInputDims = inputDims[0];
    assert(nbOutputs == 1);
    mOutputDims = outputDims[0];
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void Padding3D::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void Padding3D::detachFromContext() {}
