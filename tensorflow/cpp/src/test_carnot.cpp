#include <iostream>
#include <memory>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>

#include "cppflow/cppflow.h"
#include "cnpy.h"

int main(int argc, const char* argv[])
{
  namespace po = boost::program_options;

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
     ("help", "produce help message")
     ("use-gpu",    po::value<int>()->default_value(0), "use gpu option")
     ("model-file", po::value<std::string>(), "model file path")
     ("data-file",  po::value<std::string>(), "data file path")
     ("test-id",    po::value<int>()->default_value(0), "test id");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
  }

  std::string model_path = vm["model-file"].as<std::string>();
  std::string data_path  = vm["data-file"].as<std::string>();
  int test_id            = vm["test-id"].as<int>();

  auto data = cnpy::npy_load(data_path) ;
  auto data_dims = data.shape.size() ;
  assert(data_dims == 2) ;

  auto nrows = data.shape[0] ;
  auto ncols = data.shape[1] ;

  std::cout<<"nb rows="<<nrows<<std::endl ;
  std::cout<<"nb cols="<<ncols<<std::endl ;

  if(test_id<0 || test_id>=nrows)
  {
    std::cout<<" TEST ID not in data range"<<std::endl ;
    return 1 ;
  }

  typedef float value_type ;
  value_type pressure      = data.data<value_type>()[test_id*ncols] ;
  value_type temperature   = data.data<value_type>()[test_id*ncols + 1] ;
  value_type ref_fugacity0 = data.data<value_type>()[test_id*ncols + 2] ;
  value_type ref_fugacity1 = data.data<value_type>()[test_id*ncols + 3] ;

  int64_t ndim = 4;
  cppflow::tensor input;

  bool use_gpu = vm["use-gpu"].as<int>() == 1 ;

  float target = 0.0;
  if(use_gpu)
  {
    input = cppflow::fill({ndim}, target);
  }
  else
  {
    std::vector<float> _data(ndim, target);
    input = cppflow::tensor(_data, {ndim});
  }
  std::cout << "tensor::device(true) : " << input.device(true) << std::endl;
  std::cout << "tensor::device(false) : " << input.device(false) << std::endl;

  auto input_tensor = input.get_tensor();
  auto raw_data = static_cast<float*>(TF_TensorData(input_tensor.get()));

  std::cout<<"INPUT PRESSURE    : "<<pressure<<std::endl ;
  std::cout<<"INPUT TEMPERATURE : "<<temperature<<std::endl ;

  raw_data[0] = pressure;
  raw_data[1] = temperature;

  cppflow::model model(model_path);
  std::cout<<"LOADED MODEL OK"<<std::endl ;

  std::cout<<"TRY PREDICTION"<<std::endl ;
  auto output = model(input);
  std::cout<<"PREDICTION OK"<<std::endl ;

  auto output_tensor = output.get_tensor();
  auto raw_output_data = static_cast<float*>(TF_TensorData(output_tensor.get()));

  std::cout<<"FUGACITY 0 "<<raw_output_data[0]<<" "<<ref_fugacity0<<std::endl ;
  std::cout<<"FUGACITY 1 "<<raw_output_data[1]<<" "<<ref_fugacity1<<std::endl ;

  return 0;
}
