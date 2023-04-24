//
// Created by kadik on 18/11/2021.
//


#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <ctime>
#include <random>
#include <chrono>
#include <typeinfo>
#include <list>


// torch:
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/Parallel.h>

// memory usage
#include <sys/resource.h>
#include <unistd.h>

using namespace std;
using namespace std::chrono;

/*
long get_mem_usage(){
    struct rusage mem_usage;
    getrusage(RUSAGE_SELF, &mem_usage);
    return mem_usage.ru_maxrss;
}*/

int main(int argc, char* argv[]){

    // set default threads per core
    at::set_num_threads(1);
    at::set_num_interop_threads(1);

    // memory baseline:
    // long baseline=get_mem_usage();

    // pars args:
    string LR;
    long long size;
    if(argc==1){
        size=6400000;
        LR="../../models/LR_model.pt";
    }
    else{
        size=stoll(argv[1]);
        LR=argv[2];
    }
    //string MLR="/work/kadik/Bureau/dev/torch_tensorflow_cpp/pytorch/models/MLR_model.pt";
    // cluster:
    //string LR="/ifpengpfs/scratch/work/r11/kadik/pytorch_tensorflow_models_performance/pytorch/models/LR_model.pt"

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({ size , 1}));

    //std::cout<<inputs<<std::endl;

    // Deserialize the ScriptModule from a file using torch::jit::load().
    torch::jit::script::Module model = torch::jit::load(LR);
    cout<<"module loading ..."<<endl;

    // time_1: -------------------------------------------
    auto time_1_start = high_resolution_clock::now();

    auto output_pred = model.forward(inputs);

    // memory:
    // cout<<size<<" memory usage: "<<get_mem_usage()-baseline<<endl;

    auto time_1_stop = high_resolution_clock::now();
    //auto duration_predict = duration_cast<microseconds>(stop_predict - start_predict);
    duration<double> time_1 = time_1_stop - time_1_start ; // secondes
    cout <<size <<" time_1 took "<< time_1.count() << " secondes\n";

    // time_3: ----------------------------------------------------------------

    for(int i =2; i<25; i++) {
        auto time_4_start = high_resolution_clock::now();

        output_pred = model.forward(inputs);

        // memory :
        // cout<<size<<" "<<i<<" memory usage: "<<get_mem_usage()-baseline<<endl;

        auto time_4_stop = high_resolution_clock::now();
        //auto duration_predict = duration_cast<microseconds>(stop_predict - start_predict);
        duration<double> time_4 = time_4_stop - time_4_start; // secondes
        cout <<size<<" time_"<< i <<" took " << time_4.count() << " secondes\n";
    }


    return 0;
}

