//
// Created by kadik on 17/12/2021.
//
/*
 * lancer le programme avec totalview pour le debeuggage : mpirun -tv -np 4 ./test_LR_MPI.exe
 * */

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

// parallele:
#include <mpi.h>
#include <mkl.h>
#include <mkl_service.h>

// torch:
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/Parallel.h>

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]){

    // initialize MPI :
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // set default threads per core
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    //mkl_set_num_threads_local(1);


    long long global_size, localdata_size;
    string LR;


    if(argc==1){
        global_size=64*10000;
        localdata_size=(long long)global_size/size;
        // chemin du modèle par défaut dans le local:
        LR="../../models/LR_model.pt";

    } else  {
        global_size=stoll(argv[1]);
        localdata_size=(long long)global_size/size;
        LR=argv[2];
    }

    // Create a vector of inputs.
    float *localdata= nullptr;
    float *globaldata= nullptr;

    // Deserialize and load model: -------------------------------------------------------------------------------------
    torch::jit::script::Module model=torch::jit::load(LR);
    // local data to predict:
    vector<torch::jit::IValue> input_localdata;


    // master proc:-----------------------------------------------------------------------------------------------------
    // créer les données de test
    if(rank==0){
        // data allocation:
        //globaldata= (float*) malloc(sizeof(float)*global_size);
        globaldata= new float[global_size];
        srand(time(0));
        for (int i=0; i<localdata_size*size; ++i) {
            globaldata[i]=rand() % 100;
        }
    }

    // create local data empty vector:
    // localdata= (float*) malloc(sizeof(float)*localdata_size);
    localdata= new float[localdata_size];

    // chrono scatter data: --------------------------------------------------------------------------------------------
    auto start_scatter = high_resolution_clock::now();
    MPI_Scatter(globaldata, localdata_size , MPI_FLOAT, localdata, localdata_size , MPI_FLOAT, 0, MPI_COMM_WORLD);
    auto stop_scatter = high_resolution_clock::now();
    auto duration_scatter = duration_cast<microseconds>(stop_scatter - start_scatter);

    float f_duration_scatter=duration_scatter.count();
    //cout<<"scatter duration : "<<rank<<" : "<<f_duration_scatter<<endl;


    // chrono convertir en tensor: -------------------------------------------------------------------------------------
    auto start_convert_to_tensor = high_resolution_clock::now();

    // convertir en tensor :
    auto options = torch::TensorOptions().dtype(torch::kFloat);
    torch::Tensor tharray = torch::from_blob(localdata, {localdata_size,1}, options);
    input_localdata.push_back(tharray);

    auto stop_convert_to_tensor = high_resolution_clock::now();
    auto duration_convert_to_tensor = duration_cast<microseconds>(stop_convert_to_tensor - start_convert_to_tensor);

    float f_duration_convert_to_tensor=duration_convert_to_tensor.count();
    //cout<<"duration convert to tensor : "<<rank<<" : "<<f_duration_convert_to_tensor<<endl;


    auto output_pred=torch::from_blob(localdata, {localdata_size,1}, options);

    for(int i =2; i<8; i++) {
        output_pred = model.forward(input_localdata).toTensor();
    }
    // chrono predict:--------------------------------------------------------------------------------------------------
    auto start_predict = high_resolution_clock::now();

    //auto output_pred = model.forward(input_localdata).toTensor(); convertir en tensor
    output_pred = model.forward(input_localdata).toTensor();

    auto stop_predict= high_resolution_clock::now();

    auto duration_predict = duration_cast<microseconds>(stop_predict - start_predict);
    // cout<<"output LR: "<<output_pred<<endl;
    //duration<double> duration_predict =stop_predict -start_predict ;
    float f_duration_predict=duration_predict.count();
    //cout<<"duration predict : "<<rank<<" : "<<f_duration_predict<<endl;


    // chrono convert to std array:-------------------------------------------------------------------------------------
    auto start_convert_to_array = high_resolution_clock::now();
    // convert to std array:
    for(auto i=0; i<localdata_size; ++i){
        localdata[i]=output_pred[i].item<float>();
    }

    auto stop_convert_to_array= high_resolution_clock::now();
    auto duration_convert_to_array = duration_cast<microseconds>(stop_convert_to_array - start_convert_to_array);

    float f_duration_convert_to_array=duration_predict.count();
    //cout<<"duration convert to array : "<<rank<<" : "<<f_duration_convert_to_array<<endl;


    // chrono gather:---------------------------------------------------------------------------------------------------
    auto start_gather = high_resolution_clock::now();
    MPI_Gather(localdata, localdata_size , MPI_FLOAT, globaldata, localdata_size , MPI_FLOAT, 0, MPI_COMM_WORLD);

    auto stop_gather= high_resolution_clock::now();
    auto duration_gather = duration_cast<microseconds>(stop_gather - start_gather);

    float f_duration_gather=duration_gather.count();
    //cout<<"duration gather : "<<rank<<" : "<<f_duration_gather<<endl;


    // Reduction ( cette partie permet de récupérer le temps maximal de tout les procs)---------------------------------

    // 1. Réduction sur le temps de scatter:
    float r_duration_scatter;
    MPI_Reduce(&f_duration_scatter, &r_duration_scatter, 1, MPI_FLOAT, MPI_MAX, 0 , MPI_COMM_WORLD);

    // 2. Réduction sur le temps de convertion d'un vecteur en tensor:
    float r_duration_convert_to_tensor;
    MPI_Reduce(&f_duration_convert_to_tensor, &r_duration_convert_to_tensor, 1, MPI_FLOAT, MPI_MAX, 0 , MPI_COMM_WORLD);

    // 3. Réduction sur le temps de prédiction:
    float r_duration_predict;
    MPI_Reduce(&f_duration_predict, &r_duration_predict, 1, MPI_FLOAT, MPI_MAX, 0 , MPI_COMM_WORLD);

    // 4. Réduction sur le temps de prédiction:
    float r_duration_convert_to_array;
    MPI_Reduce(&f_duration_convert_to_array, &r_duration_convert_to_array, 1, MPI_FLOAT, MPI_MAX, 0 , MPI_COMM_WORLD);

    // 5. Réduction sur le temps de prédiction:
    float r_duration_gather;
    MPI_Reduce(&f_duration_gather, &r_duration_gather, 1, MPI_FLOAT, MPI_MAX, 0 , MPI_COMM_WORLD);

    //MPI_Allreduce(&d, &reduction_result, 1, MPI_FLOAT, MPI_SUM, 0 , MPI_COMM_WORLD);
    // cout << "reduction " << rank <<": "<<r_duration_scatter << "\n";

    // cout << "Intra-op parallelism, number of threads: " << at::get_num_threads() << endl;
    // cout << "Inter-op parallelism, number of threads: "<< at::get_num_interop_threads() << endl;

    // affichage:
    if (rank == 0) {
        /*
         * affichage des temps de réponses de chaque opération.
         */
        cout << "\nduration_scatter took " << r_duration_scatter<< " microseconds\n";
        cout << "duration_convert_to_tensor took " << r_duration_convert_to_tensor << " microseconds\n";
        cout << "duration_predict took " << r_duration_predict << " microseconds\n";
        cout << "duration_convert_to_array took " << r_duration_convert_to_array << " microseconds\n";
        cout << "duration_gather took " << r_duration_gather << " microseconds\n";
        auto total_duration=r_duration_scatter+r_duration_convert_to_tensor+r_duration_predict
                +r_duration_convert_to_array+r_duration_gather;
        cout << "total took " <<  total_duration << " microseconds\n";


        // free memorie:
        // free(globaldata);
        // free(localdata);
        delete globaldata;
        delete localdata;
    }

    // Finalize:
    MPI_Finalize();
    return 0;
}

