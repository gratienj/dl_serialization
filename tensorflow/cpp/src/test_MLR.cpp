//
// Created by kadik on 26/11/2021.
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

#include "cppflow/ops.h"
#include "cppflow/model.h"

// kadik
#include <omp.h>
//

using  namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]){

    string MLR;
    long long in_size;
    if(argc==1){
        in_size=6400000;
        MLR="/work/kadik/Bureau/dev/torch_tensorflow_cpp/tensorflow/models/MLR_model.pb";
    }
    else{
        in_size=stoll(argv[1]);
        MLR=argv[2];
    }

    vector<int64_t> size{in_size,2};
    vector<float> data;


    srand(time(0));
    for (int i=0; i<size[0]*size[1]; ++i) {
        data.push_back(rand() % 100);
    }

    // convert to tensor:
    auto input=cppflow::tensor(data,size);
    // cout<<"input: "<<input<<endl;


    //auto input = cppflow::fill({10, 1}, 1.0f);
    //std::cout<<"input: "<<input<<std::endl;
    int num_thread=omp_get_num_threads();
    cout<<"Thread rank before: "<< omp_get_num_threads() <<endl;

    //--------------

    cppflow::model model(MLR);

    cout<<"Thread rank after: "<< omp_get_num_threads() <<endl;


    // time_1: -------------------------------------------
    auto time_1_start = high_resolution_clock::now();

    auto output_pred = model(input);

    auto time_1_stop = high_resolution_clock::now();


    //--------------
    //auto duration_predict = duration_cast<microseconds>(stop_predict - start_predict);
    duration<double> time_1 = time_1_stop - time_1_start ; // secondes
    cout <<in_size <<" time_1 took "<< time_1.count() << " secondes\n";

    // time_3: ----------------------------------------------------------------

    for(int i =2; i<25; i++) {
        auto time_4_start = high_resolution_clock::now();

        output_pred  = model(input);

        auto time_4_stop = high_resolution_clock::now();
        //auto duration_predict = duration_cast<microseconds>(stop_predict - start_predict);
        duration<double> time_4 = time_4_stop - time_4_start; // secondes
        cout <<in_size<<" time_"<< i <<" took " << time_4.count() << " secondes\n";
    }

    return 0;
}