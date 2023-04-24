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

// functions:
// create 2D matrix:
double *Matrix(int rows, int cols) {
    // flatten saved matrix
    srand(time(0));
    //double * arr = (double *)malloc(rows*cols*sizeof(double));
    double* arr= new double[rows*cols];

    // fill with random data
    for (int i=0; i<rows*cols; i++)
        arr[i] = rand() % 100;

    return arr;
}

// show 2D matrix:
void show(double *arr, int rows, int cols){
    //show
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            cout<<arr[i*cols+j]<<" ";
        }
        cout<<endl;
    }
}

void free_mem(double *arr){
    delete [] arr;
}


//-------------------------------------------------
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
    string path;


    if(argc==1){
        global_size=403200;
        localdata_size=(long long)global_size/size;
        path="../../models/MLR_model.pt";

    } else  {
        global_size=stoll(argv[1]);
        localdata_size=(long long)global_size/size;
        path=argv[2];
    }

    // Create a vector of inputs.
    double *localdata= nullptr;
    double *globaldata= nullptr;
    double *gather_data= nullptr;
    int cols=2, model_call_num=10;

    /* defining a datatype for sub-matrix */
    int count=localdata_size, length=cols, stride=cols;
    MPI_Datatype sub_vec;
    MPI_Type_vector(count, length, stride, MPI_DOUBLE, &sub_vec);
    MPI_Type_commit(&sub_vec);

    // create localdata memory holder:
    localdata = Matrix(localdata_size, cols);

    // Deserialize and load model: -------------------------------------------------------------------------------------
    torch::jit::script::Module model=torch::jit::load(path);
    // local data to predict:
    vector<torch::jit::IValue> input_localdata;

    // master proc:-----------------------------------------------------------------------------------------------------
    // créer les données de test
    if(rank==0){
        // data allocation:
        srand(time(0));
        globaldata= Matrix(global_size, cols);

        /* afficher les données:
        cout<<"data :------------------------------"<<endl;
        show(globaldata, global_size, cols);
        cout<<"------------------------------------"<<endl;
        */

        //gather data:
        gather_data= Matrix(global_size, cols);

    }

    // chrono scatter data: --------------------------------------------------------------------------------------------
    auto start_scatter = high_resolution_clock::now();
    MPI_Scatter(globaldata, 1, sub_vec, localdata, 1, sub_vec, 0, MPI_COMM_WORLD);
    auto stop_scatter = high_resolution_clock::now();
    duration<double> duration_scatter = stop_scatter - start_scatter ; // secondes
    float f_duration_scatter=duration_scatter.count();
    /*show data
    cout<<"-----------------"<<rank<<"-----------------"<<endl;
    show(localdata, localdata_size, cols);
    */


    // chrono convertir en tensor: -------------------------------------------------------------------------------------
    auto start_convert_to_tensor = high_resolution_clock::now();
    // convertir en tensor :
    auto options = torch::TensorOptions().dtype(torch::kFloat);
    torch::Tensor tharray = torch::from_blob(localdata, {localdata_size,cols}, options);
    input_localdata.push_back(tharray);
    auto stop_convert_to_tensor = high_resolution_clock::now();
    duration<double> duration_convert_to_tensor = stop_convert_to_tensor - start_convert_to_tensor ; // secondes
    float f_duration_convert_to_tensor=duration_convert_to_tensor.count();



    // chrono predict:--------------------------------------------------------------------------------------------------
    vector<float> duration_predict_vec;
    auto output_pred=torch::from_blob(localdata, {localdata_size,cols}, options); // initialize
    auto start_predict= high_resolution_clock::now(); // initialize
    auto stop_predict= high_resolution_clock::now();  // initialize
    duration<double> duration_predict =stop_predict -start_predict ; // initialize
    float f_duration_prediction=0; // initialize
    for(int i =0; i<model_call_num; i++) {
        start_predict = high_resolution_clock::now(); // time start
        output_pred = model.forward(input_localdata).toTensor(); // model inference
        stop_predict= high_resolution_clock::now(); // time stop
        duration_predict =stop_predict -start_predict; // mesure time (time stop - time start)
        f_duration_prediction=duration_predict.count();
        //cout<<"i: "<< i <<", durée: "<< f_duration_prediction<<"\n";
        duration_predict_vec.push_back(f_duration_prediction); // save in vector
    }




    // chrono convert to std array:-------------------------------------------------------------------------------------
    auto start_convert_to_array = high_resolution_clock::now();
    // convert to std array:
    for(auto i=0; i<localdata_size; ++i){
        localdata[i]=output_pred[i].item<float>();
    }
    auto stop_convert_to_array= high_resolution_clock::now();
    duration<double> duration_convert_to_array= stop_convert_to_array - start_convert_to_array;
    float f_duration_convert_to_array=duration_predict.count();


    // chrono gather:---------------------------------------------------------------------------------------------------
    auto start_gather = high_resolution_clock::now();
    MPI_Gather(localdata, 1, sub_vec, gather_data , 1, sub_vec, 0, MPI_COMM_WORLD);
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


    /*free mem*/
    free_mem(localdata);

    if(rank==0){
        /*
         * affichage des temps de réponses de chaque opération.
         */
        cout << "\nduration_scatter took " << r_duration_scatter<< " secondes\n";
        cout << "duration_convert_to_tensor took " << r_duration_convert_to_tensor << " secondes\n";
        for(int i=0; i<model_call_num; i++) {
            cout << "duration_predict_" << i << " took " << r_duration_predict_vec[i] << " secondes\n";
        }
        cout << "duration_convert_to_array took " << r_duration_convert_to_array << " secondes\n";
        cout << "duration_gather took " << r_duration_gather << " secondes\n";
        auto total_duration=r_duration_scatter+r_duration_convert_to_tensor
                            +r_duration_convert_to_array+r_duration_gather;
        cout << "total hors prediction took " <<  total_duration << " secondes\n";

        // free mem:
        free_mem(globaldata);
        free_mem(gather_data);
        MPI_Type_free(&sub_vec);
    }

    // Finalize:
    MPI_Finalize();
    return 0;
}

