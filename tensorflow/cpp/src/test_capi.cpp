#include <stdio.h>
#include <tensorflow/c/c_api.h>

int main() {
  printf("Hello from TensorFlow C library version %s\n", TF_Version());
  
  TF_Graph* Graph = TF_NewGraph();
  TF_Status* Status = TF_NewStatus();

  TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
  TF_Buffer* RunOpts = NULL;

  const char* saved_model_dir = "../../../model/mnist/model/"; // Path of the model
  const char* tags = "serve"; // default model serving tag; can change in future
  int ntags = 1;

  TF_Session* Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
  if(TF_GetCode(Status) == TF_OK)
  {
        printf("TF_LoadSessionFromSavedModel OK\n");
  }
  else
  {
        printf("%s",TF_Message(Status));
  }
  return 0;
}

