#include "TFcaller.h"

TFcaller::TFcaller(std::string model_path)
{
   //Initialize the tensorflow session
   Graph = TF_NewGraph();
   Status = TF_NewStatus();

   SessionOpts = TF_NewSessionOptions();
   uint8_t intra_op_parallelism_threads = 1;
   uint8_t inter_op_parallelism_threads = 1;
   uint8_t buf[] = {0x10,intra_op_parallelism_threads, 0x28, inter_op_parallelism_threads};
   TF_SetConfig(SessionOpts, buf, sizeof(buf), Status);
   if ( TF_GetCode(Status) != TF_OK ) {
      printf("TF_SetConfig failed\n");   
   }
   //RunOpts = NULL;

   const char* saved_model_dir = model_path.c_str();
   const char* tags = "serve"; // default model serving tag; can change in future
   int ntags = 1;
   
   Session = TF_LoadSessionFromSavedModel(SessionOpts, NULL, saved_model_dir, &tags, ntags, Graph, NULL, Status);
   if ( TF_GetCode(Status) != TF_OK ) {
      fprintf(stderr, "Error : Couldn't load the Neural Net for AI wall modeling --> %s \n", TF_Message(Status));
   }
   else {
      fprintf(stderr, "(Wall Modeling) Tensorflow neural net for wall modeling successfully imported\n");
   }

   //Get informations about input and output operations
   //Running the session needs it, and they can be found by their name
   //tf v1.x saving format offered the possibility to find them automatically with the C-API
   //tf v2.x doesn't offer this possibility anymore, it must thus be donne manually using saved_model_cli command in a shell, and inform them as follows 
   input_op = {TF_GraphOperationByName(Graph, "serving_default_dense_input"), 0};
   output_op = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};

   //initialize shapes
   int64_t returned_dims_in[2];
   int64_t returned_dims_out[2];
   TF_GraphGetTensorShape(Graph, input_op, returned_dims_in, 2, Status);
   TF_GraphGetTensorShape(Graph, output_op, returned_dims_out, 2, Status);

   n_input_ai = returned_dims_in[1];
   n_output_ai = returned_dims_out[1];

   fprintf(stderr, "(Wall Modeling) Number inputs in NN: %d \n" ,n_input_ai);
   fprintf(stderr, "(Wall Modeling) Number outputs in NN: %d \n" ,n_output_ai);

   data = (float *)malloc(n_input_ai * sizeof(float));
   size_t data_size = sizeof(float) * 1 * n_input_ai;
   const int64_t dims[2] = {1, n_input_ai};
   input_tensor = TF_NewTensor(TF_FLOAT, dims, 2, data, data_size, DeallocateTensor, NULL);
   data = (float*) TF_TensorData(input_tensor);
   output_tensor = NULL;
}


TFcaller::~TFcaller() {
   free(data);
   TF_DeleteGraph(Graph);
   TF_DeleteSession(Session, Status);
   TF_DeleteSessionOptions(SessionOpts);
   TF_DeleteStatus(Status);
   //
   if (norm_param_X != NULL) delete[] norm_param_X;
   if (norm_param_Y != NULL) delete[] norm_param_Y;
}

double* TFcaller::run_ai(double* input_ai)
{
   //Declare output
   double *output_ai = new double[n_output_ai];

   //data needs to be passed as a 1D array to fill tensor
   for (int j = 0; j < n_input_ai; j++) {
      data[j] = static_cast<float>(input_ai[j]);
   }

   //Run the session
   //TF_Tensor* output_tensor = NULL;
   TF_SessionRun(Session, NULL, &input_op, &input_tensor, 1, &output_op,
                 &output_tensor, 1, nullptr, 0, nullptr, Status);

   //Write and return results
   output_data = (float *)TF_TensorData(output_tensor);
   for(int j = 0; j < n_output_ai; j++)
   {
      output_ai[j] = static_cast<double>(output_data[j]);
   }

   //free memory
   TF_DeleteTensor(output_tensor);
   output_tensor = NULL;

   return output_ai;
}

// Function to normalize input vector
double* TFcaller::normalize_input(double* state_X)
{
   // Declare normalized state
   double *state_X_norm = new double[n_input_ai];

   for(int i = 0; i < n_input_ai; i++)
      state_X_norm[i] = (state_X[i]-norm_param_X[i][0])/(sqrt(norm_param_X[i][1]));

   return state_X_norm;
}

// Function to denormalize output vector
double* TFcaller::denormalize_output(double* state_Y_norm)
{
   // Declare denormalized state
   double *state_Y = new double[n_output_ai];

   for(int i = 0; i < n_output_ai; i++)
      state_Y[i] = state_Y_norm[i]*sqrt(norm_param_Y[i][1]) + norm_param_Y[i][0];

   return state_Y;
}
