
#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <fstream>
#include <random>
#include <string>
#include <memory>


#include "graphutils.h"
#include "modelutils.h"


#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>

int main(int argc, char **argv)
{
    namespace po = boost::program_options;
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "produce help message")
    ("use-gpu", po::value<int>()->default_value(0), "use gpu option")
    ("test-load-graph", po::value<int>()->default_value(0), "test load graph")
    ("test-convert-graph-to-ptgraph", po::value<int>()->default_value(0), "test convert graph to ptgraph")
    ("test-load-model", po::value<int>()->default_value(0), "test load model")
    ("test-inference",  po::value<int>()->default_value(0), "test inference")
    ("graph-file", po::value<std::string>(), "graph file path")
    ("model-file", po::value<std::string>(), "model file path") ;


    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    bool use_gpu = vm["use-gpu"].as<int>() == 1 ;

    if(vm["test-load-graph"].as<int>() == 1)
    {
      std::string graph_path = vm["graph-file"].as<std::string>();
      std::cout<<"TEST LOAD GRAPH : "<<graph_path<<std::endl ;
      Graph graph ;
      loadFromFile(graph,graph_path) ;
      std::cout<<graph<<std::endl ;
    }

    if(vm["test-convert-graph-to-ptgraph"].as<int>() == 1)
    {
      std::string graph_path = vm["graph-file"].as<std::string>();
      std::cout<<"TEST LOAD GRAPH : "<<graph_path<<std::endl ;
      Graph graph ;
      loadFromFile(graph,graph_path) ;
      PTGraph pt_graph ;
      convertGraph2PTGraph(graph,pt_graph);
      std::cout<<pt_graph<<std::endl ;
    }

    if(vm["test-load-model"].as<int>() == 1)
    {
      std::string model_path = vm["model-file"].as<std::string>();
      std::cout<<"TEST LOAD MODEL : "<<model_path<<std::endl ;
      torch::jit::script::Module model = read_model(model_path, use_gpu);
    }

    if(vm["test-inference"].as<int>() > 0)
    {
      std::string graph_path = vm["graph-file"].as<std::string>();
      std::string model_path = vm["model-file"].as<std::string>();
      std::cout<<"TEST INFERENCE : "<<graph_path<<" "<<model_path<<std::endl ;

      Graph graph ;
      loadFromFile(graph,graph_path) ;
      std::vector<PTGraph> graphs(1) ;
      convertGraph2PTGraph(graph,graphs[0]);

      torch::jit::script::Module model = read_model(model_path, use_gpu);
      int opt = vm["test-inference"].as<int>() ;
      auto predictions = infer(model, graphs, use_gpu,opt);
      for(auto& pred : predictions)
      {
         std::cout<<"PRED : "<<std::endl ;
         std::cout<<pred<<std::endl ;
      }
    }

    return 0;
}