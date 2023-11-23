/*
 * TRTEngine.h
 *
 *  Created on: 29 sept. 2023
 *      Author: gratienj
 */
#pragma once

class TRTEngine
{
public:
    TRTEngine(std::string const& onnx_model_path,
    		  int size,
			  int nrows,
			  int ncols)
        : m_onnx_model_path(onnx_model_path)
        , m_size(size)
        , m_nrows(nrows)
        , m_ncols(ncols)
        , mRuntime(nullptr)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer(std::vector<float> const& input, std::vector<float>& output, int batch_size);
    bool infer(std::vector<double> const& input, std::vector<double>& output, int batch_size);

private:

    std::string m_onnx_model_path ;
    int         m_size = 0 ;
    int         m_nrows = 0 ;
    int         m_ncols = 0 ;

#ifdef USE_TENSORRT
    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    samplesCommon::ManagedBuffer mInput{};          //!< Host and device buffers for the input.
    samplesCommon::ManagedBuffer mOutput{};

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                          SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
						  SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                          SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(std::vector<float> const& inputs);
    bool processInput(std::vector<double> const& inputs);

    template<typename ValueT>
    bool _processInput(std::vector<ValueT> const& inputs);

    template<typename ValueT>
    bool _infer(std::vector<ValueT> const& input, std::vector<ValueT>& output, int batch_size);
#endif

};

