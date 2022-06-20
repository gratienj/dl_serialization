#ifndef TF_CALLER_H
#define TF_CALLER_H
#include "tensorflow/c/c_api.h"
#include "math.h"
#include <iostream>

using std::cout;
using std::endl;

class TFcaller
{
    public:
        TFcaller();
        TFcaller(std::string model_path);
        ~TFcaller();

        double* run_ai(double* input_ai);
        double* normalize_input(double* state_X);
        double* denormalize_output(double* state_Y_norm);
        static void DeallocateTensor(void* data, size_t length, void* argument) {};
       
        int n_input_ai;
        int n_output_ai;
 
    private:
        TF_Graph* Graph;
        TF_Status* Status;
        TF_SessionOptions* SessionOpts;
        TF_Session* Session;

        TF_Output input_op;
        TF_Output output_op;

        TF_Tensor* input_tensor;
        TF_Tensor* output_tensor;

        double** norm_param_X = NULL;
        double** norm_param_Y = NULL;

        float* data = NULL;
        float* output_data = NULL;
};
#endif