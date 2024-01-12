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
    TRTEngine(std::string const& onnx_model_path, int batch_size)
    : m_onnx_model_path(onnx_model_path)
    , m_batch_size(batch_size)
    , mRuntime(nullptr)
    , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build(int nb_vertices, int nb_edges);

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer(std::vector<float> const& x,
               std::vector<int> const& edge_index,
               std::vector<float> const& edge_attr,
               std::vector<float> const& y,
               std::vector<float>& output,
               int nb_vertices,
               int nb_edges);

    bool infer(std::vector<double> const& x,
               std::vector<int> const& edge_index,
               std::vector<double> const& edge_attr,
               std::vector<double> const& y,
               std::vector<double>& output,
               int nb_vertices,
               int nb_edges);
private:

    std::string m_onnx_model_path ;
    int         m_batch_size = 0 ;

#ifdef USE_TENSORRT
    nvinfer1::Dims mInputDims0;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mInputDims1;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mInputDims2;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mInputDims3;  //!< The dimensions of the input to the network.
    samplesCommon::ManagedBuffer mInput0{};          //!< Host and device buffers for the input.
    samplesCommon::ManagedBuffer mInput1{};          //!< Host and device buffers for the input.
    samplesCommon::ManagedBuffer mInput2{};          //!< Host and device buffers for the input.
    samplesCommon::ManagedBuffer mInput3{};          //!< Host and device buffers for the input.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    samplesCommon::ManagedBuffer mOutput{};

    SampleUniquePtr<IRuntime> mRuntime{};
    //std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
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
    bool processInput(std::vector<float> const& inputs,samplesCommon::ManagedBuffer& mInput);
    bool processInput(std::vector<double> const& inputs,samplesCommon::ManagedBuffer& mInput);
    bool processInput(std::vector<int> const& input,samplesCommon::ManagedBuffer& mInput);

    template<typename ValueT>
    bool _processInput(std::vector<ValueT> const& inputs,
                       samplesCommon::ManagedBuffer& mInput);

    template<typename ValueT>
    bool _infer(std::vector<ValueT> const& x,
                std::vector<int> const& edge_index,
                std::vector<ValueT> const& edge_attr,
                std::vector<ValueT> const& y,
                std::vector<ValueT>& output,
                int nb_vertices,
                int nb_edges);
#endif

};

