//
// Created by kadik on 08/03/2022.
//

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


#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>

//#include <filesystem>
#include <boost/filesystem.hpp>
#include "cppflow/ops.h"
#include "cppflow/model.h"



using  namespace std;
using namespace std::chrono;


// vider cache:
void micro_benchmark_clear_cache() {
    std::vector<int> clear = std::vector<int>();
    clear.resize(1000 * 1000 * 1000, 42);
    for (size_t i = 0; i < clear.size(); i++) {
        clear[i] += 1;
    }
    clear.resize(0);
}



int main(int argc, char* argv[]){

    namespace po = boost::program_options;
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "produce help message")
    ("use-gpu",         po::value<int>()->default_value(0),     "use gpu option")
    ("nb-samples",      po::value<int>()->default_value(64000), "nb samples")
    ("nb-features",     po::value<int>()->default_value(1),     "nb features")
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
    string model_path    = vm["model-file"].as<std::string>();
    int nb_samples       = vm["nb-samples"].as<int>();
    int features         = vm["nb-features"].as<int>();
    int model_call_num   = vm["nb-calls"].as<int>();


    vector<int64_t> size{nb_samples,features};
    vector<float> data;


    srand(time(0));
    for (int i=0; i<size[0]*size[1]; ++i) {
        data.push_back(rand() % 100);
    }

    // convert to tensor:
    auto input=cppflow::tensor(data,size);
    // cout<<"input: "<<input<<endl;


    //--------------

    cppflow::model model(model_path);


    // time_1: -------------------------------------------
    auto time_1_start = high_resolution_clock::now();
    auto output_pred = model(input);
    auto time_1_stop = high_resolution_clock::now();

    //--------------
    //auto duration_predict = duration_cast<microseconds>(stop_predict - start_predict);
    duration<double> time_1 = time_1_stop - time_1_start ; // secondes
    cout <<nb_samples <<" time_1 took "<< time_1.count() << " secondes\n";

    // time_3: ----------------------------------------------------------------
    // vider le cache:
    micro_benchmark_clear_cache();


    for(int i =2; i<10; i++) {
        auto time_4_start = high_resolution_clock::now();
        output_pred  = model(input);
        auto time_4_stop = high_resolution_clock::now();
        //auto duration_predict = duration_cast<microseconds>(stop_predict - start_predict);
        duration<double> time_4 = time_4_stop - time_4_start; // secondes
        cout <<nb_samples<<" time_"<< i <<" took " << time_4.count() << " secondes\n";

        // vider le cache:
        micro_benchmark_clear_cache();
    }

    return 0;
}