#include "padding3DPlugin.h"
#include "plugin.h"
#include "padding3D.h"
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

Padding3D::Padding3D(const Dims& prePadding, const Dims& postPadding) {
    mPrePadding = prePadding;
    mPostPadding = postPadding;
}

int Padding3D::getNbOutputs() const
{
    return 1;
}

//Dims Padding3D::getOutputDimensions(int index, const Dims* inputDims, int nbInputs)
nvinfer1::DimsExprs Padding3D::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    assert(nbInputs == 1);
    nvinfer1::DimsExprs const& input = inputs[0];
    assert(outputIndex == 0);
    nvinfer1::DimsExprs output;
    output.nbDims = input.nbDims;
    //last 3dim
    auto pad_dim = output.nbDims - 3;
    for (int d = 0; d < input.nbDims; ++d) {
        output.d[d] = input.d[d]; 
        if (output.d[d]->isConstant()) {
            if (d >= pad_dim) {
                auto ind = 3 - output.nbDims + d;
                assert(ind >= 0);
                output.d[d] = exprBuilder.constant(output.d[d]->getConstantValue() + mPrePadding.d[ind] + mPostPadding.d[ind]);
            }
            std::cout <<"smq getOutput: pad after:" << output.d[d]->getConstantValue() << std::endl;
        } else {
            std::cout <<"smq getOutput dynamic:-1" << std::endl;
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

size_t Padding3D::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const
{
    return 0;
}

size_t Padding3D::getSerializationSize() const
{
    // dimensions: 3 * 2
    return sizeof(int) * (5 + 3 * 2);
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

IPluginV2DynamicExt* Padding3D::clone() const
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

//bool Padding3D::supportsFormat(DataType type, PluginFormat format) const
bool Padding3D::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    assert(nbInputs == 1);
    assert(nbOutputs == 1);
    const PluginTensorDesc& desc = inOut[pos];
    return (desc.type == DataType::kFLOAT && desc.dims.nbDims == 5);
}

int Padding3D::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) {
    int batch_size = inputDesc->dims.d[0];
    cuPad::cuPadding3D(outputs[0], mOutputDims, inputs[0], mInputDims, batch_size, mInputDims, mPrePadding, mPostPadding, 0, stream);
    std::cout << "enqueue= " << mOutputDims.d[2] << " " << mOutputDims.d[3] << " " << mOutputDims.d[4] << std::endl;
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
//bool Padding3D::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
//{
//    return false;
//}
//
//// Return true if plugin can use input that is broadcast across batch without replication.
//bool Padding3D::canBroadcastInputAcrossBatch(int inputIndex) const
//{
//    return false;
//}

// Configure the layer with input and output data types.
//void Padding3D::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
//    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
//    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
void Padding3D::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, 
        int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {
    assert(nbInputs == 1);
    mInputDims = in[0].desc.dims;
    assert(nbOutputs == 1);
    mOutputDims = out[0].desc.dims;
}

//// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
//void Padding3D::attachToContext(
//    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
//{
//}
//
//// Detach the plugin object from its execution context.
//void Padding3D::detachFromContext() {}

