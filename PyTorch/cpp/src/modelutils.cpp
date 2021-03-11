
#include "graphutils.h"
#include "modelutils.h"

// Predict

torch::Tensor __predict_basic(torch::jit::script::Module model, PTGraph const& graph) {

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(graph.m_x);
  inputs.push_back(graph.m_edge_index);

  // Execute the model and turn its output into a tensor.
  torch::NoGradGuard no_grad;
  torch::Tensor output = model.forward(inputs).toTensor();

  torch::DeviceType cpu_device_type = torch::kCPU;
  torch::Device cpu_device(cpu_device_type);
  output = output.to(cpu_device);

  return output;
}

torch::Tensor __predict(torch::jit::script::Module model, PTGraph const& graph) {

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(graph.m_batch) ;
  inputs.push_back(graph.m_x);
  inputs.push_back(graph.m_edge_index);
  inputs.push_back(graph.m_edge_attr);

  // Execute the model and turn its output into a tensor.
  torch::NoGradGuard no_grad;
  torch::Tensor output = model.forward(inputs).toTensor();

  torch::DeviceType cpu_device_type = torch::kCPU;
  torch::Device cpu_device(cpu_device_type);
  output = output.to(cpu_device);

  return output;
}


// 1. Read model
torch::jit::script::Module read_model(std::string model_path, bool usegpu) {

  std::cout<<"LOAD SCRIPT MODEL"<<std::endl ;
  torch::jit::script::Module model = torch::jit::load(model_path);

  if (usegpu) {
      torch::DeviceType gpu_device_type = torch::kCUDA;
      torch::Device gpu_device(gpu_device_type);

      model.to(gpu_device);
  } else {
      torch::DeviceType cpu_device_type = torch::kCPU;
      torch::Device cpu_device(cpu_device_type);
      model.to(cpu_device);
  }
  return model;
}

// 2. Forward
std::vector<PredictionT<PTGraph::value_type> >
forward_basic(std::vector<PTGraph>& graphs,
              torch::jit::script::Module model)
{
  std::vector<PredictionT<PTGraph::value_type>> predictions ;
  torch::DeviceType cpu_device_type = torch::kCPU;
  torch::Device cpu_device(cpu_device_type);
  for(auto& g : graphs)
  {
    g.m_x.to(cpu_device) ;
    g.m_edge_index.to(cpu_device) ;
    std::cout<<"TRY PREDICT"<<std::endl ;
    torch::Tensor output = __predict_basic(model, g);

    output = output.to(cpu_device);

    int ndim = output.ndimension();
    assert(ndim == 2);

    torch::ArrayRef<int64_t> sizes = output.sizes();

    PredictionT<PTGraph::value_type> prediction ;
    prediction.m_dim0 = sizes[0] ;
    prediction.m_dim1 = sizes[1] ;

    std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes[1]<<std::endl ;
    //assert(n_samples == 1);
    prediction.m_values = std::vector<PTGraph::value_type>(output.data_ptr<PTGraph::value_type>(),
                                   output.data_ptr<PTGraph::value_type>() + (sizes[0] * sizes[1]));
    predictions.push_back(std::move(prediction)) ;
  }
  return predictions;
}

// 2. Forward
std::vector<PredictionT<PTGraph::value_type> >
forward(std::vector<PTGraph>& graphs,
        torch::jit::script::Module model,
        bool usegpu)
{
  std::vector<PredictionT<PTGraph::value_type>> predictions ;

  if (usegpu)
  {
      torch::DeviceType gpu_device_type = torch::kCUDA;
      torch::Device gpu_device(gpu_device_type);
      for(auto& g : graphs)
      {
        g.m_batch.to(gpu_device) ;
        g.m_x.to(gpu_device) ;
        g.m_edge_index.to(gpu_device) ;
        g.m_edge_attr.to(gpu_device) ;
        torch::Tensor output = __predict(model, g);


        torch::DeviceType cpu_device_type = torch::kCPU;
        torch::Device cpu_device(cpu_device_type);
        output = output.to(cpu_device);
        int ndim = output.ndimension();
        assert(ndim == 2);
        torch::ArrayRef<int64_t> sizes = output.sizes();
        PredictionT<PTGraph::value_type> prediction ;
        prediction.m_dim0 = sizes[0] ;
        prediction.m_dim1 = sizes[1] ;

        std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes[1]<<std::endl ;
        prediction.m_values = std::vector<PTGraph::value_type>(output.data_ptr<PTGraph::value_type>(),
                                   output.data_ptr<PTGraph::value_type>() + (sizes[0] * sizes[1]));
        predictions.push_back(std::move(prediction)) ;
      }

      //tensor = tensor.to(gpu_device);
  }
  else
  {
      torch::DeviceType cpu_device_type = torch::kCPU;
      torch::Device cpu_device(cpu_device_type);
      for(auto& g : graphs)
      {
        g.m_batch.to(cpu_device) ;
        g.m_x.to(cpu_device) ;
        g.m_edge_index.to(cpu_device) ;
        g.m_edge_attr.to(cpu_device) ;
        torch::Tensor output = __predict(model, g);

        output = output.to(cpu_device);
        int ndim = output.ndimension();
        assert(ndim == 2);
        torch::ArrayRef<int64_t> sizes = output.sizes();
        PredictionT<PTGraph::value_type> prediction ;
        prediction.m_dim0 = sizes[0] ;
        prediction.m_dim1 = sizes[1] ;
        std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes[1]<<std::endl ;

        prediction.m_values = std::vector<PTGraph::value_type>(output.data_ptr<PTGraph::value_type>(),
                                   output.data_ptr<PTGraph::value_type>() + (sizes[0] * sizes[1]));
        predictions.push_back(std::move(prediction)) ;
      }
  }
  return predictions;
}

std::vector<PredictionT<PTGraph::value_type> >
infer(torch::jit::script::Module model, std::vector<PTGraph>& graphs,bool usegpu,int opt)
{
    // Forward
    switch(opt)
    {
    case 1 :
         return forward(graphs, model, usegpu);
    case 2 :
         return forward_basic(graphs, model);
    default :
         return forward(graphs, model, usegpu);
    }
}
