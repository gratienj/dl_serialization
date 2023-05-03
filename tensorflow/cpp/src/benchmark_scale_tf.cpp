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

    string MLR;
    long long nb_samples;
    int features;
    if(argc==1){
        MLR="/work/kadik/Bureau/dev/torch_tensorflow_cpp/tensorflow/CPU/models/MLR_model.pb";
        nb_samples=100;
        features=1;
    }
    else{
        MLR=argv[1];
        nb_samples=stoll(argv[2]);
        features=stoi(argv[3]);

    }

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

    cppflow::model model(MLR);


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