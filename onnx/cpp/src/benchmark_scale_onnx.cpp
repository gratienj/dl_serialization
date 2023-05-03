//
// Created by kadik on 04/07/2022.
//

//
// Created by kadik on 14/06/2022.
//

/*
 * lancer le programme avec totalview pour le debeuggage : mpirun -tv -np 4 ./test_LR_MPI.exe
 **/

// std:
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>
#include <memory>
#include <ctime>
#include <random>
#include <chrono>
#include <typeinfo>
#include <list>

//BOOST
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>

//#include <filesystem>
#include <boost/filesystem.hpp>

//ONNX
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

using namespace std;
using namespace std::chrono;

// functions:
// create 2D matrix:
void Matrix(int rows, int cols, std::vector<float>& arr) {
    srand(time(0));
    float v;
    for (int i=0; i<rows*cols; i++){
        v=rand()%100;
        arr.push_back(v);
    }
}


// show 2D matrix:
void show(std::vector<float>& v, int cols){
    //show
    for(size_t i=0; i<v.size(); i++){
        if(i%cols==0) cout<<endl;
        cout<<v[i]<<" ";
    }
    cout<<endl;
}

template <typename T>
ostream& operator<<(ostream& o, const vector<T>& v){
    o<<endl;
    for(size_t i=0; i<v.size(); i++){
        o<<v[i]<<" ";
    }
    o<<endl;
    return o;
}

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}


//-------------------------------------------------
int main(int argc, char* argv[]){

    namespace po = boost::program_options;
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "produce help message")
    ("use-gpu",         po::value<int>()->default_value(0),     "use gpu option")
    ("nb-samples",      po::value<int>()->default_value(64000), "nb samples")
    ("nb-features",     po::value<int>()->default_value(1),     "nb features")
    ("nb-output-cols",  po::value<int>()->default_value(1),     "nb output cols")
    ("nb-calls",        po::value<int>()->default_value(10),    "nb calls")
    ("model-file",      po::value<std::string>(),               "model file path") ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    bool use_gpu         = vm["use-gpu"].as<int>() == 1 ;
    string module_path   = vm["model-file"].as<std::string>();
    int global_size      = vm["nb-samples"].as<int>();
    int features         = vm["nb-features"].as<int>();
    int output_cols      = vm["nb-output-cols"].as<int>();
    int model_call_num   = vm["nb-calls"].as<int>();

    // Create a vector of inputs.
    vector<float> input_localdata(global_size*features);
    vector<float> output_localdata(global_size*features);




    /* defining a datatype for sub-matrix */
    srand(time(0));
    Matrix(global_size, features, input_localdata);


    // onnxruntime setup
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING);
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel( GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // session:
    Ort::Session session(env, module_path.c_str(), sessionOptions);

    // text:
    // Ort::Session gives access to input and output information:
    // - counts
    // - name
    // - shape and type
    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    //std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    //std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;

    // generate data:
    auto inputName = session.GetInputNameAllocated(0, allocator);
    //std::cout << "Input Name: " << inputName << std::endl;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    //std::cout << "Input Type: " << inputType << std::endl;

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    //std::cout << "Input Dimensions: " << inputDims << std::endl;

    auto outputName = session.GetOutputNameAllocated(0, allocator);
    //std::cout << "Output Name: " << outputName << std::endl;

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();


    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    //std::cout << "Output Type: " << outputType << std::endl;

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    //std::cout << "Output Dimensions: " << outputDims << std::endl;

    // input tensor size:
    //size_t inputTensorSize = vectorProduct(inputDims);
    size_t inputTensorSize = global_size*features;
    std::vector<float> inputTensorValues(inputTensorSize);
    //std::vector<int> inputDims={localdata_size, features};
    //std::vector<int> outputDims={localdata_size, features};

    int outputTensorSize=global_size*features;


    // serving names:
    const char* input_names[] = {inputName.get()};
    char* output_names[] = {outputName.get()};

    // onnx tensors:
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    // batch size:
    inputDims[0]=global_size;
    outputDims[0]=global_size;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu( OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, input_localdata.data(), inputTensorSize , inputDims.data(),
            inputDims.size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, output_localdata.data(), outputTensorSize,
            outputDims.data(), outputDims.size()));

    /*
    // Measure latency
    int numTests{100};
    std::chrono::steady_clock::time_point begin =
            std::chrono::steady_clock::now();
    for (int i = 0; i < numTests; i++)
    {
        session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                    inputTensors.data(), 1, outputNames.data(),
                    outputTensors.data(), 1);
    }
    std::chrono::steady_clock::time_point end =
            std::chrono::steady_clock::now();
    std::cout << "Minimum Inference Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() /
                 static_cast<float>(numTests)
              << " ms" << std::endl;
    */

    // chrono predict:--------------------------------------------------------------------------------------------------
    auto start_predict= high_resolution_clock::now(); // initialize
    auto stop_predict= high_resolution_clock::now();  // initialize
    duration<double> duration_predict =stop_predict -start_predict ; // initialize
    float f_duration_prediction=0; // initialize
    for(int i =0; i<model_call_num; i++) {
        start_predict = high_resolution_clock::now(); // time start

        session.Run(Ort::RunOptions{nullptr}, input_names,
                    inputTensors.data(), 1, output_names,
                    outputTensors.data(), 1);  // model inference

        stop_predict= high_resolution_clock::now(); // time stop
        duration_predict =stop_predict -start_predict; // mesure time (time stop - time start)
        f_duration_prediction=duration_predict.count();
        cout <<global_size<<" duration_predict_" << i << " took " << f_duration_prediction << " secondes\n";

    }



    return 0;
}


