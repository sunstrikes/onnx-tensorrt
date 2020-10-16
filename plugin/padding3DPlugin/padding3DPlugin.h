#pragma once
#include <cassert>
#include <cuda_runtime_api.h>
#include <string.h>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"

namespace nvinfer1
{
namespace plugin
{
class Padding3D : public IPluginV2DynamicExt
{
public:
    Padding3D(const Dims& prePadding, const Dims& postPadding);

    Padding3D(const void* data, size_t length);

    ~Padding3D() override = default;
    
    //v2DynamicExt
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

    //V2 methods
    int getNbOutputs() const override;

    //Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    void destroy() override;

    //size_t getWorkspaceSize(int) const override;

    //int enqueue(
    //    int batch_size, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    //bool supportsFormat(DataType type, PluginFormat format) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    IPluginV2DynamicExt* clone() const override;

    void setPluginNamespace(const char* libNamespace) override;

    const char* getPluginNamespace() const override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    //bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    //bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    //void attachToContext(
    //    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

    //void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    //    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    //    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;

    //void detachFromContext() override;
protected:
    // To prevent compiler warnings.
    using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::configurePlugin;
    using nvinfer1::IPluginV2DynamicExt::enqueue;
    using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
    using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
    using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::supportsFormat;
private:
    Dims mPrePadding;
    Dims mPostPadding;
    Dims mInputDims;
    Dims mOutputDims;
    std::string mNameSpace;
};

class Padding3DPluginCreator : public IPluginCreator
{
public:
    Padding3DPluginCreator();

    ~Padding3DPluginCreator(){};

    void setPluginNamespace(const char* libNamespace) override {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const override {
        return mNamespace.c_str();
    }

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* data, size_t length) override;

private:
    static PluginFieldCollection mFC;
    std::string mNamespace;
    //Dims prePadding;
    //Dims postPadding;
    static std::vector<PluginField> mPluginAttributes;
};
} // namespace plugin
} // namespace nvinfer1

