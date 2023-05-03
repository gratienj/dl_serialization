//
// Created by kadik on 24/02/2022.
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

// parallele:
#include <mpi.h>
#include <mkl.h>
#include <mkl_service.h>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>

//#include <filesystem>
#include <boost/filesystem.hpp>
// torch:
#include "cppflow/ops.h"
#include "cppflow/model.h"

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
void show(std::vector<float>& arr, int rows, int cols){
    //show
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            cout<<arr[i*cols+j]<<" ";
        }
        cout<<endl;
    }
}

void show(std::vector<std::vector<float>>& arr, int rows, int cols){
    //show
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            cout<<arr[i][j]<<" ";
        }
        cout<<endl;
    }
}

// vider cache:
void micro_benchmark_clear_cache() {
    std::vector<int> clear = std::vector<int>();
    clear.resize(1000 * 1000 * 1000, 42);
    for (size_t i = 0; i < clear.size(); i++) {
        clear[i] += 1;
    }
    clear.resize(0);
}


//-------------------------------------------------
int main(int argc, char* argv[]){

    // initialize MPI :
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    namespace po = boost::program_options;
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "produce help message")
    ("use-gpu",         po::value<int>()->default_value(0),     "use gpu option")
    ("nb-samples",      po::value<int>()->default_value(40320), "nb samples")
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
    int global_size      = vm["nb-samples"].as<int>();
    int cols             = vm["nb-features"].as<int>();
    int out_size=1 ;
    int model_call_num   = vm["nb-calls"].as<int>();

    long long localdata_size (long long)global_size/size;


    // Create a vector of inputs.
    vector<float> localdata;
    vector<float> globaldata;
    vector<float> gather_data;


    /* defining a datatype for sub-matrix */
    int count=localdata_size, length=cols, stride=cols;
    MPI_Datatype sub_vec;
    MPI_Type_vector(count, length, stride, MPI_FLOAT, &sub_vec);
    MPI_Type_commit(&sub_vec);

    // create localdata memory holder:
    Matrix(localdata_size, cols, localdata);

    // Deserialize and load model: -------------------------------------------------------------------------------------
    cppflow::model model(model_path);

    // master proc:-----------------------------------------------------------------------------------------------------
    // créer les données de test
    if(rank==0){
        // data allocation:
        srand(time(0));
        Matrix(global_size, cols, globaldata);

        //afficher les données:
        /*
        cout<<"data :------------------------------"<<endl;
        show(globaldata, global_size, cols);
        cout<<"------------------------------------"<<endl;
        */


        //gather data:
        Matrix(global_size, out_size, gather_data);

    }

    // chrono scatter data: --------------------------------------------------------------------------------------------
    auto start_scatter = high_resolution_clock::now();
    MPI_Scatter(globaldata.data(), 1, sub_vec, localdata.data(), 1, sub_vec, 0, MPI_COMM_WORLD);
    auto stop_scatter = high_resolution_clock::now();
    duration<double> duration_scatter = stop_scatter - start_scatter ; // secondes
    float f_duration_scatter=duration_scatter.count();
    //show data
    /*
    cout<<"-----------------"<<rank<<"-----------------"<<endl;
    show(localdata, localdata_size, cols);
    */


    // chrono convertir en tensor: -------------------------------------------------------------------------------------
    vector<int64_t> size_data{localdata_size,cols};

    auto start_convert_to_tensor = high_resolution_clock::now();
    // convertir en tensor :
    auto input=cppflow::tensor(localdata,size_data);
    auto stop_convert_to_tensor = high_resolution_clock::now();
    duration<double> duration_convert_to_tensor = stop_convert_to_tensor - start_convert_to_tensor ; // secondes
    float f_duration_convert_to_tensor=duration_convert_to_tensor.count();
    //cout<<"tensor : "<<input<<endl;

    // chrono predict:--------------------------------------------------------------------------------------------------
    vector<float> duration_predict_vec;
    cppflow::tensor output_pred; // initialize
    auto start_predict= high_resolution_clock::now(); // initialize
    auto stop_predict= high_resolution_clock::now();  // initialize
    duration<double> duration_predict =stop_predict -start_predict ; // initialize
    double f_duration_prediction=0; // initialize
    for(int i =0; i<model_call_num; i++) {
        start_predict = high_resolution_clock::now(); // time start
        output_pred = model(input); // model inference
        stop_predict= high_resolution_clock::now(); // time stop
        duration_predict =stop_predict -start_predict; // mesure time (time stop - time start)
        f_duration_prediction=(float)duration_predict.count();
        //cout<<"i: "<< i <<", durée: "<< f_duration_prediction<<"\n";
        duration_predict_vec.push_back(f_duration_prediction); // save in vector

        // vider le cache:
        micro_benchmark_clear_cache();
    }

    // chrono tensor -> std::vector:------------------------------------------------------------------------------------
    auto start_convert_to_array = high_resolution_clock::now();
    // convert to std array:
    std::vector<float> out;
    out=output_pred.get_data<float>();
    auto stop_convert_to_array= high_resolution_clock::now();
    duration<double> duration_convert_to_array= stop_convert_to_array - start_convert_to_array;
    float f_duration_convert_to_array=duration_predict.count();

    /*
    cout<<"-----------------"<<rank<<"-----------------"<<endl;
    cout<<"predict size: "<<localdata_size<<"x"<<cols<<endl;
    show(out, localdata_size, out_size);
    */




    // chrono gather:---------------------------------------------------------------------------------------------------
    auto start_gather = high_resolution_clock::now();
    MPI_Gather(out.data(), localdata_size, MPI_FLOAT, gather_data.data() , localdata_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    auto stop_gather= high_resolution_clock::now();
    duration<double> duration_gather = stop_gather - start_gather;
    float f_duration_gather=duration_gather.count();


    // Reduction ( cette partie permet de récupérer le temps maximal de tout les procs)---------------------------------
    /*
     * @breaf trouver le temps maximal pour chaque process pour chaque étape de calcul
     * */

    // 1. Réduction sur le temps de scatter:
    float r_duration_scatter;
    MPI_Reduce(&f_duration_scatter, &r_duration_scatter, 1, MPI_FLOAT, MPI_MAX, 0 , MPI_COMM_WORLD);

    // 2. Réduction sur le temps de convertion d'un vecteur en tensor:
    float r_duration_convert_to_tensor;
    MPI_Reduce(&f_duration_convert_to_tensor, &r_duration_convert_to_tensor, 1, MPI_FLOAT, MPI_MAX, 0 , MPI_COMM_WORLD);

    // 3. Réduction sur le temps de prédiction:
    vector<float> r_duration_predict_vec;
    r_duration_predict_vec.resize(duration_predict_vec.size());  // store reduction times of model calls (mem holder)
    MPI_Reduce(duration_predict_vec.data(), r_duration_predict_vec.data(), model_call_num , MPI_FLOAT, MPI_MAX, 0 , MPI_COMM_WORLD);

    // 4. Réduction sur le temps de conversion to array:
    float r_duration_convert_to_array;
    MPI_Reduce(&f_duration_convert_to_array, &r_duration_convert_to_array, 1, MPI_FLOAT, MPI_MAX, 0 , MPI_COMM_WORLD);

    // 5. Réduction sur le temps de gather:
    float r_duration_gather;
    MPI_Reduce(&f_duration_gather, &r_duration_gather, 1, MPI_FLOAT, MPI_MAX, 0 , MPI_COMM_WORLD);


    if(rank==0){
        /*
         * affichage des temps de réponses de chaque opération.
         */
        /*
        cout<<"-----------------"<<rank<<"-----------------"<<endl;
        cout<<"gather: "<<endl;
        show(gather_data, global_size, out_size);
        */

        cout << "duration_scatter took " << r_duration_scatter<< " secondes\n";
        cout << "duration_convert_to_tensor took " << r_duration_convert_to_tensor << " secondes\n";
        for(int i=0; i<model_call_num; i++) {
            cout <<size<<" duration_predict_" << i << " took " << r_duration_predict_vec[i] << " secondes\n";
        }
        cout << "duration_convert_to_array took " << r_duration_convert_to_array << " secondes\n";
        cout << "duration_gather took " << r_duration_gather << " secondes\n";
        auto total_duration=r_duration_scatter+r_duration_convert_to_tensor+r_duration_convert_to_array
                            +r_duration_gather;
        cout << "total hors prediction took " <<  total_duration << " secondes\n";

        // free mem:
        cout<<"##########finished##########"<<endl;
        MPI_Type_free(&sub_vec);
    }

    // Finalize:
    MPI_Finalize();
    return 0;
}


