/*
 * TRTEngine.cpp
 *
 *  Created on: 29 sept. 2023
 *      Author: gratienj
 */

#ifdef USE_TENSORRT
#include "utils/9.2/argsParser.h"
#include "utils/9.2/buffers.h"
#include "utils/9.2/common.h"
#include "utils/9.2/logger.h"
#include "utils/9.2/parserOnnxConfig.h"

#include "NvInfer.h"

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;
#endif

#include "internal/TRTEngine.h"


#ifdef USE_TENSORRT
//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool TRTEngine::build(int nb_vertices, int nb_edges)
{
std::cout<<"CREATE BUILDER"<<std::endl ;
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    std::cout<<"CREATE NETWORK"<<std::endl ;
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    std::cout<<"CREATE CONFIG"<<std::endl ;
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    std::cout<<"CREATE PARSER"<<std::endl ;
    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    std::cout<<"BUILD NETWORK"<<std::endl ;
    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    std::cout<<"NB INPUTS : "<<network->getNbInputs()<<std::endl ;
    auto input0 = network->getInput(0) ;
    auto dims0 = network->getInput(0)->getDimensions() ;
    std::cout<<" INPUT[0] : "<<input0->getName()<<" DIMS "<<dims0<<std::endl ;
    auto input1 = network->getInput(1) ;
    auto dims1 = network->getInput(1)->getDimensions() ;
    std::cout<<" INPUT[1] : "<<input1->getName()<<" DIMS "<<dims1<<std::endl ;
    auto input2 = network->getInput(2) ;
    auto dims2 = network->getInput(2)->getDimensions() ;
    std::cout<<" INPUT[2] : "<<input2->getName()<<" DIMS "<<dims2<<std::endl ;
    auto input3 = network->getInput(3) ;
    auto dims3 = network->getInput(3)->getDimensions() ;
    std::cout<<" INPUT[3] : "<<input3->getName()<<" DIMS "<<dims3<<std::endl ;

    auto output = network->getOutput(0) ;
    auto out_dims0 = output->getDimensions() ;
    std::cout<<"NB OUTPUTS : "<<network->getNbOutputs()<<std::endl ;
    std::cout<<" OUTPUT[0] : "<<network->getOutput(0)->getName()<<" DIMS "<<out_dims0<<std::endl ;


    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    auto profile = builder->createOptimizationProfile();
    if(dims0.nbDims == 2)
    {
      profile->setDimensions(input0->getName(), OptProfileSelector::kMIN, Dims2{nb_vertices, dims0.d[1]});
      profile->setDimensions(input0->getName(), OptProfileSelector::kOPT, Dims2{nb_vertices, dims0.d[1]});
      profile->setDimensions(input0->getName(), OptProfileSelector::kMAX, Dims2{nb_vertices, dims0.d[1]});
    }
    if(dims1.nbDims == 2)
    {
      profile->setDimensions(input1->getName(), OptProfileSelector::kMIN, Dims2{dims1.d[0], nb_edges});
      profile->setDimensions(input1->getName(), OptProfileSelector::kOPT, Dims2{dims1.d[0], nb_edges});
      profile->setDimensions(input1->getName(), OptProfileSelector::kMAX, Dims2{dims1.d[0], nb_edges});
    }
    if(dims2.nbDims == 2)
    {
      profile->setDimensions(input2->getName(), OptProfileSelector::kMIN, Dims2{nb_edges, dims2.d[1]});
      profile->setDimensions(input2->getName(), OptProfileSelector::kOPT, Dims2{nb_edges, dims2.d[1]});
      profile->setDimensions(input2->getName(), OptProfileSelector::kMAX, Dims2{nb_edges, dims2.d[1]});
    }
    if(dims3.nbDims == 2)
    {
      profile->setDimensions(input3->getName(), OptProfileSelector::kMIN, Dims2{nb_vertices, dims3.d[1]});
      profile->setDimensions(input3->getName(), OptProfileSelector::kOPT, Dims2{nb_vertices, dims3.d[1]});
      profile->setDimensions(input3->getName(), OptProfileSelector::kMAX, Dims2{nb_vertices, dims3.d[1]});
    }

    if(out_dims0.nbDims == 2)
    {
      profile->setDimensions(output->getName(), OptProfileSelector::kMIN, Dims2{nb_vertices, out_dims0.d[1]});
      profile->setDimensions(output->getName(), OptProfileSelector::kOPT, Dims2{nb_vertices, out_dims0.d[1]});
      profile->setDimensions(output->getName(), OptProfileSelector::kMAX, Dims2{nb_vertices, out_dims0.d[1]});
    }

    config->addOptimizationProfile(profile);

    std::cout<<"BUILD SERIALIZED NETWORK"<<std::endl ;
    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        std::cout<<"BUILD SERIALIZED NETWORK FAILED"<<std::endl ;
        return false;
    }

    std::cout<<"CREATE RUNTIME"<<std::endl ;
    if (!mRuntime)
    {
        mRuntime = SampleUniquePtr<IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    }
    //mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime)
    {
        std::cout<<"CREATE RUNTIME FAILED"<<std::endl ;
        return false;
    }

    std::cout<<"CREATE ENGINE"<<std::endl ;
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        std::cout<<"CREATE ENGINE FAILED"<<std::endl ;
        return false;
    }

    ASSERT(network->getNbInputs() == 4);
    mInputDims0 = network->getInput(0)->getDimensions();
    mInputDims1 = network->getInput(1)->getDimensions();
    mInputDims2 = network->getInput(2)->getDimensions();
    mInputDims3 = network->getInput(3)->getDimensions();

    std::cout<<"INPUT DIMS : "<<network->getNbInputs()<<std::endl ;
    std::cout<<"             "<<mInputDims0.nbDims<<std::endl ;
    std::cout<<"             "<<mInputDims1.nbDims<<std::endl ;
    std::cout<<"             "<<mInputDims2.nbDims<<std::endl ;
    std::cout<<"             "<<mInputDims3.nbDims<<std::endl ;
    //ASSERT(mInputDims.nbDims == 2);

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    //ASSERT(mOutputDims.nbDims == 2);

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool TRTEngine::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                                       SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
   SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                                       SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(m_onnx_model_path.c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }
    /*
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0F, 127.0F);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
    */

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
template<typename ValueT>
bool TRTEngine::_infer(std::vector<ValueT> const& x,
                       std::vector<int> const& edge_index,
                       std::vector<ValueT> const& edge_attr,
                       std::vector<ValueT> const& y,
                       std::vector<ValueT>& outputs,
                       int nb_vertices,
                       int nb_edges)
{
    std::cout<<"INFER"<<std::endl ;

    // Create RAII buffer manager object
    //samplesCommon::BufferManager buffers(mEngine,m_size);

    std::cout<<"CREATE CONTEXT"<<std::endl ;
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    auto dims0 = mEngine->getBindingDimensions(0) ;
    std::cout<<"PROCESS INPUT : "<<x.size()<<" "<<dims0<<std::endl ;
    if(dims0.nbDims == 2)
    {
      Dims2 inputDims{nb_vertices,dims0.d[1]} ;
      mInput0.hostBuffer.resize(inputDims) ;
      if (!processInput(x,mInput0))
      {
          return false;
      }

      mInput0.deviceBuffer.resize(inputDims);
      CHECK(cudaMemcpy(
          mInput0.deviceBuffer.data(), mInput0.hostBuffer.data(), mInput0.hostBuffer.nbBytes(), cudaMemcpyHostToDevice));

      CHECK_RETURN_W_MSG(context->setBindingDimensions(0, inputDims), false, "Invalid binding dimensions.");
    }

    auto dims1 = mEngine->getBindingDimensions(1) ;
    std::cout<<"PROCESS INPUT : "<<edge_index.size()<<" "<<dims1<<std::endl ;
    if(dims1.nbDims == 2)
    {
      Dims2 inputDims{dims1.d[0],nb_edges} ;
      mInput1.hostBuffer.resize(inputDims) ;
      if (!processInput(edge_index,mInput1))
      {
          return false;
      }

      mInput1.deviceBuffer.resize(inputDims);
      CHECK(cudaMemcpy(
          mInput1.deviceBuffer.data(), mInput1.hostBuffer.data(), mInput1.hostBuffer.nbBytes(), cudaMemcpyHostToDevice));

      CHECK_RETURN_W_MSG(context->setBindingDimensions(1, inputDims), false, "Invalid binding dimensions.");
    }

    auto dims2 = mEngine->getBindingDimensions(2) ;
    std::cout<<"PROCESS INPUT : "<<edge_attr.size()<<" "<<dims2<<std::endl ;
    if(dims2.nbDims == 2)
    {
      Dims2 inputDims{nb_edges,dims2.d[1]} ;
      mInput2.hostBuffer.resize(inputDims) ;
      if (!processInput(edge_attr,mInput2))
      {
          return false;
      }

      mInput2.deviceBuffer.resize(inputDims);
      CHECK(cudaMemcpy(
          mInput2.deviceBuffer.data(), mInput2.hostBuffer.data(), mInput2.hostBuffer.nbBytes(), cudaMemcpyHostToDevice));

      CHECK_RETURN_W_MSG(context->setBindingDimensions(2, inputDims), false, "Invalid binding dimensions.");
    }

    auto dims3 = mEngine->getBindingDimensions(3) ;
    std::cout<<"PROCESS INPUT : "<<edge_attr.size()<<" "<<dims3<<std::endl ;
    if(dims3.nbDims == 2)
    {
      Dims2 inputDims{nb_vertices,dims3.d[1]} ;
      mInput2.hostBuffer.resize(inputDims) ;
      if (!processInput(y,mInput3))
      {
          return false;
      }

      mInput3.deviceBuffer.resize(inputDims);
      CHECK(cudaMemcpy(
          mInput3.deviceBuffer.data(), mInput3.hostBuffer.data(), mInput3.hostBuffer.nbBytes(), cudaMemcpyHostToDevice));

      CHECK_RETURN_W_MSG(context->setBindingDimensions(3, inputDims), false, "Invalid binding dimensions.");
    }


    if (!context->allInputDimensionsSpecified())
    {
        std::cout<<"INPUT DIM NOT SPECIFIED"<<std::endl ;
        return false;
    }


    auto outdims = mEngine->getBindingDimensions(1) ;
    if(outdims.nbDims == 2)
    {
      Dims2 outputDims{nb_vertices,outdims.d[1]} ;
      std::cout<<"PROCESS OUTPUT : "<<outdims<<" "<<outputDims<<std::endl ;
      mOutput.hostBuffer.resize(outputDims) ;
      mOutput.deviceBuffer.resize(outputDims);
      //CHECK_RETURN_W_MSG(context->setBindingDimensions(0, outputDims), false, "Invalid binding dimensions.");

    }


    // Read the input data into the managed buffers
    //ASSERT(mParams.inputTensorNames.size() == 1);

    // Memcpy from host input buffers to device input buffers
    //buffers.copyInputToDevice();


    std::vector<void*> bindings = {mInput0.deviceBuffer.data(),
                                   mInput1.deviceBuffer.data(),
                                   mInput2.deviceBuffer.data(),
                                   mInput3.deviceBuffer.data(),
                                   mOutput.deviceBuffer.data()};

    bool status = context->executeV2(bindings.data());
    if (status)
    {
        std::cout<<"TensorRT INFERENCE OK"<<std::endl ;
        CHECK(cudaMemcpy(mOutput.hostBuffer.data(), mOutput.deviceBuffer.data(), mOutput.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost));

        float* hostDataBuffer = static_cast<float*>(mOutput.hostBuffer.data());
        for (int i = 0; i < outputs.size(); i++)
        {
            outputs[i] = hostDataBuffer[i];
            //std::cout<<"OUTPUT["<<i<<"]"<<outputs[i]<<std::endl ;
        }
        return true ;
    }
    else
    {
        std::cout<<"TensorRT INFERENCE FAILED"<<std::endl ;
        return false;
    }
}


bool TRTEngine::infer(std::vector<float> const& x,
                      std::vector<int> const& edge_index,
                      std::vector<float> const& edge_attr,
                      std::vector<float> const& y,
                      std::vector<float>& outputs,
                      int nb_vertices,
                      int nb_edges)
{
  return _infer(x,edge_index,edge_attr,y,outputs,nb_vertices,nb_edges) ;
}

bool TRTEngine::infer(std::vector<double> const& x,
                      std::vector<int> const& edge_index,
                      std::vector<double> const& edge_attr,
                      std::vector<double> const& y,
                      std::vector<double>& outputs,
                      int nb_vertices,
                      int nb_edges)
{
  return _infer(x,edge_index,edge_attr,y,outputs,nb_vertices,nb_edges) ;
}


//!
//! \brief Reads the input and stores the result in a managed buffer
//!
template<typename ValueT>
bool TRTEngine::_processInput(std::vector<ValueT> const& inputs,
                              samplesCommon::ManagedBuffer& mInput)
{
    std::cout<<"PROCESS INPUT"<<std::endl ;
    ValueT* hostDataBuffer = static_cast<ValueT*>(mInput.hostBuffer.data());
    for (int i = 0; i < inputs.size(); i++)
    {
        hostDataBuffer[i] = inputs[i];
    }

    return true;
}

bool TRTEngine::processInput(std::vector<float> const& inputs,
                             samplesCommon::ManagedBuffer& mInput)
{
  return _processInput(inputs, mInput) ;
}

bool TRTEngine::processInput(std::vector<double> const& inputs,
                             samplesCommon::ManagedBuffer& mInput)
{
  return _processInput(inputs, mInput) ;
}

bool TRTEngine::processInput(std::vector<int> const& inputs,
                             samplesCommon::ManagedBuffer& mInput)
{
  return _processInput(inputs, mInput) ;
}
#endif

