
#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <fstream>
#include <random>
#include <string>
#include <sstream>
#include <memory>


#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>

//#include <filesystem>
#include <boost/filesystem.hpp>

#include <torch/script.h>
#include <torch/serialize/tensor.h>
#include <torch/serialize.h>

#include "cnpy.h"

int main(int argc, char **argv)
{
    namespace po = boost::program_options;
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "produce help message")
    ("use-gpu",         po::value<int>()->default_value(0), "use gpu option")
    ("test-load-model", po::value<int>()->default_value(0), "test load model")
    ("test-inference",  po::value<int>()->default_value(0), "test inference")
    ("test-data-id",    po::value<int>()->default_value(0), "test data id")
    ("batch-size",      po::value<int>()->default_value(1), "batch size")
    ("nb-comp",         po::value<int>()->default_value(2), "nb compo")
    ("data-file",       po::value<std::string>(),           "data file path")
    ("model-file",      po::value<std::string>(),           "model file path") ;


    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    bool use_gpu = vm["use-gpu"].as<int>() == 1 ;

    if(vm["test-load-model"].as<int>() == 1)
    {
      std::string model_path = vm["model-file"].as<std::string>();
      std::cout<<"TEST LOAD MODEL : "<<model_path<<std::endl ;
      torch::jit::script::Module model = torch::jit::load(model_path);
    }

    if(vm["test-inference"].as<int>() > 0 )
    {
       std::string data_path = vm["data-file"].as<std::string>();
       auto data = cnpy::npy_load(data_path) ;
       auto data_dims = data.shape.size() ;
       assert(data_dims == 2) ;

       auto nrows = data.shape[0] ;
       auto ncols = data.shape[1] ;


       int nb_comp = vm["nb-comp"].as<int>() ;
       std::cout<<"nb rows  ="<<nrows<<std::endl ;
       std::cout<<"nb cols  ="<<ncols<<std::endl ;
       std::cout<<"nb comps ="<<nb_comp<<std::endl ;

       assert(ncols >= 2 + nb_comp) ;

       int test_data_id = vm["test-data-id"].as<int>() ;
       int batch_size   = vm["batch-size"].as<int>() ;

       typedef double value_type ;
       std::size_t tensor_size = 2 + nb_comp ;
       std::vector<double> x(batch_size*tensor_size) ;
       std::cout<<"TEST DATA ID : "<<test_data_id<<std::endl ;
       int offset = 0 ;
       for(int i=0;i<batch_size;++i)
       {
           std::cout<<"[P, T, Z ] : [";
           for(int ic=0;ic<tensor_size;++ic)
           {
              x[offset+ic] = data.data<value_type>()[ic*nrows+test_data_id+i] ;
              std::cout<<x[offset + ic]<<",";
           }
           std::cout<<"]"<<std::endl ;
           offset += tensor_size ;
       }

       std::string model_path = vm["model-file"].as<std::string>();
       std::cout<<"TEST INFERENCE : "<<model_path<<std::endl ;
       torch::jit::script::Module model = torch::jit::load(model_path);

       torch::DeviceType device_type = use_gpu ? torch::kCUDA : torch::kCPU;
       torch::Device device(device_type);
       model.to(device);

       std::vector<torch::jit::IValue> inputs;
       std::vector<int64_t> dims = { batch_size, tensor_size};
       torch::TensorOptions options(torch::kFloat64);
       torch::Tensor input = torch::from_blob(x.data(), torch::IntList(dims), options).clone().to(device);
       inputs.push_back(input) ;
       std::cout<<"FORWARD : ";
       auto outputs = model.forward(inputs).toTuple();
       std::cout<<"AFTER FORWARD"<<std::endl ;

       torch::Device cpu_device(torch::kCPU);
       //auto cpu_outputs = outputs.to(cpu_device);
       {
           auto out = outputs->elements()[0].toTensor().to(cpu_device);
           torch::ArrayRef<int64_t> sizes = out.sizes();
           std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes.size()<<std::endl ;
           auto r = std::vector<bool>(out.data_ptr<bool>(),out.data_ptr<bool>() + sizes[0]);
           for(int i=0;i<sizes[0];++i)
           {
             std::cout<<"UNSTABLE["<<i<<"]="<<r[i]<<std::endl ;
           }
       }
       {
           auto out = outputs->elements()[1].toTensor().to(cpu_device);
           //auto out_acc = out.accessor<double,1>() ;
           //std::cout<<"SIZE : "<<out_acc.size(0)<<std::endl ;
           torch::ArrayRef<int64_t> sizes = out.sizes();
           std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes[1]<<" "<<sizes.size()<<std::endl ;
           auto r = std::vector<double>(out.data_ptr<double>(),out.data_ptr<double>() + sizes[0]);
           for(int i=0;i<sizes[0];++i)
           {
             std::cout<<"THETA_V["<<i<<"]="<<r[i]<<std::endl ;
           }
       }
       {
           auto out = outputs->elements()[2].toTensor().to(cpu_device);
           //auto out_acc = out.accessor<double,2>() ;
           //std::cout<<"SIZE : "<<out_acc.size(0)<<std::endl ;
           torch::ArrayRef<int64_t> sizes = out.sizes();
           std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes[1]<<" "<<sizes.size()<<std::endl ;
           auto r = std::vector<double>(out.data_ptr<double>(),out.data_ptr<double>() + sizes[0]*sizes[1]);
           for(int i=0;i<sizes[0];++i)
           {
             std::cout<<"XI["<<i<<"]=";
             for(int j=0;j<sizes[1];++j)
                std::cout<<r[i*sizes[1]+j]<<",";
              std::cout<<std::endl ;
           }
       }
       {
           auto out = outputs->elements()[3].toTensor().to(cpu_device);
           torch::ArrayRef<int64_t> sizes = out.sizes();
           std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes[1]<<" "<<sizes.size()<<std::endl ;
           auto r = std::vector<double>(out.data_ptr<double>(),out.data_ptr<double>() + sizes[0]*sizes[1]);
           for(int i=0;i<sizes[0];++i)
           {
             std::cout<<"YI["<<i<<"]=";
             for(int j=0;j<sizes[1];++j)
                std::cout<<r[i*sizes[1]+j]<<",";
              std::cout<<std::endl ;
           }
       }
       {
           auto out = outputs->elements()[4].toTensor().to(cpu_device);
           torch::ArrayRef<int64_t> sizes = out.sizes();
           std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes[1]<<" "<<sizes.size()<<std::endl ;
           auto r = std::vector<double>(out.data_ptr<double>(),out.data_ptr<double>() + sizes[0]*sizes[1]);
           for(int i=0;i<sizes[0];++i)
           {
             std::cout<<"KI["<<i<<"]=";
             for(int j=0;j<sizes[1];++j)
                std::cout<<r[i*sizes[1]+j]<<",";
              std::cout<<std::endl ;
           }
       }
    }


    return 0;
}
