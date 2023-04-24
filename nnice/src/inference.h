#ifndef INFERENCE_H
#define INFERENCE_H
#include "Eigen/Core"
#include "hdf5.h"
#include "rapidjson/document.h"
#include "vector"
#include <iostream>

using std::cout;
using std::endl;

// use typedefs for future ease for changing data types like : float to double
typedef Eigen::MatrixXd Matrix;
typedef Eigen::RowVectorXd RowVector;
typedef void (*activ_ptr)(RowVector& input);

class Inferator
{
    public:
        Inferator();
        Inferator(std::string model_path);
        ~Inferator();

        double* run_ai(double* input_ai);

        double* normalize_input(double* state_X);
        double* denormalize_output(double* state_Y_norm);

        double* compute_wall_func_ai_inputs(double* wall_values, int n_var);
        void calc_inv_transition_matrix(double* P_inv_matrix, double* P_matrix, double* NN_var, double* local_norm);
        void NN_var_vector_proj(double* NN_var, double* NN_var_local, double* P_inv_matrix);
        void NN_var_tensor_proj(double* NN_var_tensor, double* NN_var_tensor_local, double* P_inv_matrix, double* P_matrix);
        
        // functions should be static, no need to have 1 per object
        static void reLU(RowVector& input) {input = input.cwiseMax(0);};
        static void tanh(RowVector& input) {input = input.array().tanh();};
        static void id(RowVector& input) {};

        int n_input_ai;
        int n_output_ai;
 
    private:
        std::string read_kernel_bias_and_jsondata(std::string path);
        void read_activations(std::string path);
        
        // HDF5 reader routine
        void add_bias_from_dataset(hid_t h5_dset);
        void add_weight_from_dataset(hid_t h5_dset);

        // function for forward propagation of data
        void propagateForward(RowVector& input);

        std::vector<RowVector, Eigen::aligned_allocator<RowVector>> neuronLayers; // stores the different layers of out network
        std::vector<Matrix, Eigen::aligned_allocator<Matrix>> weights; // the connection weights itself
        std::vector<RowVector, Eigen::aligned_allocator<RowVector>> bias; // stores the different layers of out network
        std::vector<activ_ptr> activationFunctions;
        std::vector<uint> topology;

        double** norm_param_X = NULL;
        double** norm_param_Y = NULL;
};
#endif

