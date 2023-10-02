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

#include "utils/argsParser.h"
#include "utils/buffers.h"
#include "utils/common.h"
#include "utils/logger.h"
#include "utils/parserOnnxConfig.h"
#include "NvInfer.h"

using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

#include "TRTEngine.h"

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
    vector<float> input_localdata;
    input_localdata.reserve(global_size*features);
    vector<float> output_localdata;
    output_localdata.reserve(global_size*features);




    /* defining a datatype for sub-matrix */
    srand(time(0));
    Matrix(global_size, features, input_localdata);



    TRTEngine engine(module_path,global_size,features,output_cols);

    std::cout << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;

    bool ok = engine.build() ;
    if(!ok)
    {
        std::cerr<<"BUILD FAILED"<<std::endl ;;
    }

    auto start_predict= high_resolution_clock::now(); // initialize
	auto stop_predict= high_resolution_clock::now();  // initialize
	duration<double> duration_predict =stop_predict -start_predict ; // initialize
	float f_duration_prediction=0; // initialize
	for(int i =0; i<model_call_num; i++) {
		start_predict = high_resolution_clock::now(); // time start

	    bool infer_status = engine.infer(input_localdata,output_localdata) ;
	    if(!infer_status)
	    {
	        std::cerr<<"INFER FAILED"<<std::endl ;;
	    }


		stop_predict= high_resolution_clock::now(); // time stop
		duration_predict =stop_predict -start_predict; // mesure time (time stop - time start)
		f_duration_prediction=duration_predict.count();
		cout <<global_size<<" duration_predict_" << i << " took " << f_duration_prediction << " secondes\n";

	}

    return 0;
}


