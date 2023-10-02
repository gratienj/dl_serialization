/*
 * TRTEngine.cpp
 *
 *  Created on: 29 sept. 2023
 *      Author: gratienj
 */

#include "utils/argsParser.h"
#include "utils/buffers.h"
#include "utils/common.h"
#include "utils/logger.h"
#include "utils/parserOnnxConfig.h"

#include "NvInfer.h"

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

#include "TRTEngine.h"


//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool TRTEngine::build()
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
    auto input = network->getInput(0) ;
    auto dims = network->getInput(0)->getDimensions() ;
    std::cout<<"NB INPUTS : "<<network->getNbInputs()<<std::endl ;
    std::cout<<" INPUT[0] : "<<input->getName()<<" DIMS "<<dims<<std::endl ;

    auto output = network->getOutput(0) ;
    std::cout<<"NB OUTPUTS : "<<network->getNbOutputs()<<std::endl ;
    std::cout<<" OUTPUT[0] : "<<network->getOutput(0)->getName()<<" DIMS "<<output->getDimensions()<<std::endl ;


    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    auto profile = builder->createOptimizationProfile();
    if(dims.nbDims == 2)
    {
      profile->setDimensions(input->getName(), OptProfileSelector::kMIN, Dims2{1,      dims.d[1]});
      profile->setDimensions(input->getName(), OptProfileSelector::kOPT, Dims2{m_size, dims.d[1]});
      profile->setDimensions(input->getName(), OptProfileSelector::kMAX, Dims2{m_size, dims.d[1]});

      profile->setDimensions(output->getName(), OptProfileSelector::kMIN, Dims2{1,      dims.d[1]});
      profile->setDimensions(output->getName(), OptProfileSelector::kOPT, Dims2{m_size, dims.d[1]});
      profile->setDimensions(output->getName(), OptProfileSelector::kMAX, Dims2{m_size, dims.d[1]});
    }
    if(dims.nbDims == 3)
    {
      profile->setDimensions(input->getName(), OptProfileSelector::kMIN, Dims3{1,      dims.d[1], dims.d[2]});
      profile->setDimensions(input->getName(), OptProfileSelector::kOPT, Dims3{m_size, dims.d[1], dims.d[2]});
      profile->setDimensions(input->getName(), OptProfileSelector::kMAX, Dims3{m_size, dims.d[1], dims.d[2]});

      profile->setDimensions(output->getName(), OptProfileSelector::kMIN, Dims3{1,      dims.d[1], dims.d[2]});
      profile->setDimensions(output->getName(), OptProfileSelector::kOPT, Dims3{m_size, dims.d[1], dims.d[2]});
      profile->setDimensions(output->getName(), OptProfileSelector::kMAX, Dims3{m_size, dims.d[1], dims.d[2]});
    }


    config->addOptimizationProfile(profile);

    std::cout<<"BUILD SERIALIZED NETWORK"<<std::endl ;
    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    std::cout<<"CREATE RUNTIME"<<std::endl ;
    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime)
    {
        return false;
    }

    std::cout<<"CREATE ENGINE"<<std::endl ;
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    std::cout<<"INPUT DIMS : "<<network->getNbInputs()<<" "<<mInputDims.nbDims<<std::endl ;
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
bool TRTEngine::infer(std::vector<float> const& inputs, std::vector<float>& outputs)
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

    auto dims = mEngine->getBindingDimensions(0) ;
	std::cout<<"PROCESS INPUT : "<<inputs.size()<<" "<<dims<<std::endl ;

    if(dims.nbDims == 2)
    {
      Dims2 inputDims{m_size,dims.d[1]} ;
      mInput.hostBuffer.resize(inputDims) ;
      if (!processInput(inputs))
      {
          return false;
      }

      mInput.deviceBuffer.resize(inputDims);
      CHECK(cudaMemcpy(
          mInput.deviceBuffer.data(), mInput.hostBuffer.data(), mInput.hostBuffer.nbBytes(), cudaMemcpyHostToDevice));

      CHECK_RETURN_W_MSG(context->setBindingDimensions(0, inputDims), false, "Invalid binding dimensions.");

      if (!context->allInputDimensionsSpecified())
	  {
			std::cout<<"INPUT DIM NOT SPECIFIED"<<std::endl ;
			return false;
	  }


    }
    if(dims.nbDims == 3)
    {
      Dims3 inputDims{m_size,dims.d[1],dims.d[2]} ;
      mInput.hostBuffer.resize(inputDims) ;
      if (!processInput(inputs))
      {
          return false;
      }

      mInput.deviceBuffer.resize(inputDims);
      CHECK(cudaMemcpy(
          mInput.deviceBuffer.data(), mInput.hostBuffer.data(), mInput.hostBuffer.nbBytes(), cudaMemcpyHostToDevice));

      CHECK_RETURN_W_MSG(context->setBindingDimensions(0, inputDims), false, "Invalid binding dimensions.");

      if (!context->allInputDimensionsSpecified())
	  {
			std::cout<<"INPUT DIM NOT SPECIFIED"<<std::endl ;
			return false;
	  }
    }


    auto outdims = mEngine->getBindingDimensions(1) ;
    if(outdims.nbDims == 2)
    {
		Dims2 outputDims{m_size,outdims.d[1]} ;
		std::cout<<"PROCESS OUTPUT : "<<outdims<<" "<<outputDims<<std::endl ;
		mOutput.hostBuffer.resize(outputDims) ;
		mOutput.deviceBuffer.resize(outputDims);
	    //CHECK_RETURN_W_MSG(context->setBindingDimensions(0, outputDims), false, "Invalid binding dimensions.");

    }

    if(outdims.nbDims == 3)
    {
		Dims3 outputDims{m_size,outdims.d[1],outdims.d[2]} ;
		std::cout<<"PROCESS OUTPUT : "<<outdims<<" "<<outputDims<<std::endl ;
		mOutput.hostBuffer.resize(outputDims) ;
		mOutput.deviceBuffer.resize(outputDims);
	    //CHECK_RETURN_W_MSG(context->setBindingDimensions(1, outputDims), false, "Invalid binding dimensions.");

    }





    // Read the input data into the managed buffers
    //ASSERT(mParams.inputTensorNames.size() == 1);

    // Memcpy from host input buffers to device input buffers
    //buffers.copyInputToDevice();


    std::vector<void*> bindings = {mInput.deviceBuffer.data(), mOutput.deviceBuffer.data()};

    bool status = context->executeV2(bindings.data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    //buffers.copyOutputToHost();

    CHECK(cudaMemcpy(
    		mOutput.hostBuffer.data(), mOutput.deviceBuffer.data(), mOutput.deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost));


    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool TRTEngine::processInput(std::vector<float> const& inputs)
{
	std::cout<<"PROCESS INPUT"<<std::endl ;
    //const int inputH = mInputDims.d[2];
    //const int inputW = mInputDims.d[3];

    /*
    // Read a random digit file
    srand(unsigned(time(nullptr)));
    std::vector<uint8_t> fileData(inputH * inputW);
    mNumber = rand() % 10;
    readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    // Print an ascii representation
    sample::gLogInfo << "Input:" << std::endl;
    for (int i = 0; i < inputH * inputW; i++)
    {
        sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    }
    sample::gLogInfo << std::endl;
    */

    float* hostDataBuffer = static_cast<float*>(mInput.hostBuffer.data());
    for (int i = 0; i < inputs.size(); i++)
    {
        hostDataBuffer[i] = inputs[i];
    }

    return true;
}

