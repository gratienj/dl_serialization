#include "graph.h"
#include "internal/graphutils.h"
#include "internal/modelutils.h"

namespace ml4cfd {
  template<typename ValueT>
  struct PredictionT
  {
      typedef ValueT value_type ;
      int m_dim0 = 0;
      int m_dim1 = 0;
      std::vector<value_type> m_values ;
  } ;
}

// Predict

template<typename value_type, typename index_type>
torch::Tensor __predict_basic(torch::jit::script::Module model, ml4cfd::PTGraphT<value_type,index_type>& graph, int nb_args,torch::Device device)
{

  std::vector<torch::jit::IValue> inputs;
  if(nb_args>0)
  {
    graph.m_x = graph.m_x.to(device) ;
    inputs.push_back(graph.m_x);
  }
  if(nb_args>1)
  {
    graph.m_edge_index = graph.m_edge_index.to(device) ;
    inputs.push_back(graph.m_edge_index);
  }
  if(nb_args>2)
  {
    graph.m_edge_attr = graph.m_edge_attr.to(device) ;
    inputs.push_back(graph.m_edge_attr);
  }

  // Execute the model and turn its output into a tensor.
  torch::NoGradGuard no_grad;
  model.to(device) ;
  torch::Tensor output = model.forward(inputs).toTensor();
  return output;
}

template<typename value_type, typename index_type>
torch::Tensor __predict(torch::jit::script::Module model, ml4cfd::PTGraphT<value_type,index_type>& graph) {

  std::vector<torch::jit::IValue> inputs;
  //inputs.push_back(graph.m_batch) ;
  inputs.push_back(graph.m_x);
  inputs.push_back(graph.m_edge_index);
  inputs.push_back(graph.m_edge_attr);
  inputs.push_back(graph.m_y);

  // Execute the model and turn its output into a tensor.
  torch::NoGradGuard no_grad;
  //std::cout<<"MODEL forward "<<inputs.size()<<std::endl ;
  torch::Tensor output = model.forward(inputs).toTensor();
  //std::cout<<"after MODEL forward"<<std::endl ;

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
template<typename value_type, typename index_type>
std::vector<ml4cfd::PredictionT<value_type> >
basic_forwardT(std::vector<ml4cfd::PTGraphT<value_type,index_type>>& graphs,
              torch::jit::script::Module model,
              int nb_args,
              bool use_gpu)
{
  std::vector<ml4cfd::PredictionT<value_type>> predictions ;
  torch::DeviceType cpu_device_type = torch::kCPU;
  torch::Device cpu_device(cpu_device_type);
  torch::DeviceType device_type = use_gpu ? torch::kCUDA : torch::kCPU;
  torch::Device device(device_type);
  for(auto& g : graphs)
  {
    torch::Tensor output = __predict_basic(model, g,nb_args,device);
    output = output.to(cpu_device);

    int ndim = output.ndimension();
    assert(ndim == 2);

    torch::ArrayRef<int64_t> sizes = output.sizes();

    ml4cfd::PredictionT<value_type> prediction ;
    prediction.m_dim0 = sizes[0] ;
    prediction.m_dim1 = sizes[1] ;

    std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes[1]<<std::endl ;
    //assert(n_samples == 1);
    prediction.m_values = std::vector<value_type>(output.data_ptr<value_type>(),
                                   output.data_ptr<value_type>() + (sizes[0] * sizes[1]));
    predictions.push_back(std::move(prediction)) ;
  }
  return predictions;
}

// 2. Forward
template<typename value_type, typename index_type>
std::vector<ml4cfd::PredictionT<value_type> >
forwardT(std::vector<ml4cfd::PTGraphT<value_type,index_type>>& graphs,
        torch::jit::script::Module model,
        bool usegpu,
        int batch_size)
{
  //std::cout<<"FORWARD : NB GRAPHS : "<<graphs.size()<<std::endl ;
  std::vector<ml4cfd::PredictionT<value_type>> predictions(graphs.size()) ;

  if (usegpu)
  {
      torch::DeviceType gpu_device_type = torch::kCUDA;
      torch::Device gpu_device(gpu_device_type);

      torch::DeviceType cpu_device_type = torch::kCPU;
      torch::Device cpu_device(cpu_device_type);
      std::size_t icount = 0 ;

      for(auto& g : graphs)
      {
        g.m_x          = g.m_x.to(gpu_device_type) ;
        g.m_edge_index = g.m_edge_index.to(gpu_device_type) ;
        g.m_edge_attr  = g.m_edge_attr.to(gpu_device_type) ;
        g.m_y          = g.m_y.to(gpu_device_type) ;

        torch::Tensor output = __predict(model, g);

        output = output.to(cpu_device);
        int ndim = output.ndimension();
        assert(ndim == 2);
        torch::ArrayRef<int64_t> sizes = output.sizes();
        //ml4cfd::PredictionT<value_type> prediction ;
        auto& prediction = predictions[icount++] ;
        prediction.m_dim0 = sizes[0] ;
        prediction.m_dim1 = sizes[1] ;

        prediction.m_values = std::vector<value_type>(output.data_ptr<value_type>(),
                                   output.data_ptr<value_type>() + (sizes[0] * sizes[1]));
      }
  }
  else
  {
      torch::DeviceType cpu_device_type = torch::kCPU;
      torch::Device cpu_device(cpu_device_type);
      std::size_t icount = 0 ;
      for(auto& g : graphs)
      {
        //std::cout<<"COMPUTE GRAPH["<<icount<<"]"<<std::endl ;
        torch::Tensor output = __predict(model, g);

        output = output.to(cpu_device);
        int ndim = output.ndimension();
        assert(ndim == 2);
        torch::ArrayRef<int64_t> sizes = output.sizes();

        auto& prediction = predictions[icount++] ;
        prediction.m_dim0 = sizes[0] ;
        prediction.m_dim1 = sizes[1] ;

        prediction.m_values = std::vector<value_type>(output.data_ptr<value_type>(),
                                   output.data_ptr<value_type>() + (sizes[0] * sizes[1]));
      }
  }
  return predictions;
}

std::vector<ml4cfd::PredictionT<ml4cfd::PTGraphT<double,int64_t>::value_type> >
infer(torch::jit::script::Module model, std::vector<ml4cfd::PTGraphT<double,int64_t>>& graphs,bool usegpu,int nb_args,int batch_size)
{
    // Forward
    if(nb_args==0)
      return forwardT<double,int64_t>(graphs, model, usegpu,batch_size);
    else
      return basic_forwardT<double,int64_t>(graphs, model, nb_args, usegpu);
}


std::vector<ml4cfd::PredictionT<ml4cfd::PTGraphT<float,int64_t>::value_type> >
infer(torch::jit::script::Module model, std::vector<ml4cfd::PTGraphT<float,int64_t>>& graphs,bool usegpu,int nb_args,int batch_size)
{
    // Forward
    if(nb_args==0)
      return forwardT<float,int64_t>(graphs, model, usegpu,batch_size);
    else
      return basic_forwardT<float,int64_t>(graphs, model, nb_args, usegpu);
}
