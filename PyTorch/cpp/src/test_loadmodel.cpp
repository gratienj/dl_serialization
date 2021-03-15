#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>

int main(int argc, const char* argv[])
{
  namespace po = boost::program_options;

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("use-gpu", po::value<int>()->default_value(0), "use gpu option")
    ("model-file", po::value<std::string>(), "model file path") ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
  }

  std::string model_path  = vm["model-file"].as<std::string>();
  bool usegpu             = vm["use-gpu"].as<int>() == 1 ;
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(model_path);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";
}