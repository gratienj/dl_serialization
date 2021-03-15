
#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <fstream>
#include <random>
#include <string>
#include <sstream>
#include <memory>


#include "graphutils.h"
#include "modelutils.h"


#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>

//#include <filesystem>
#include <boost/filesystem.hpp>


int main(int argc, char **argv)
{
    namespace po = boost::program_options;
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "produce help message")
    ("use-gpu",         po::value<int>()->default_value(0), "use gpu option")
    ("test-load-graph", po::value<int>()->default_value(0), "test load graph")
    ("test-convert-graph-to-ptgraph", po::value<int>()->default_value(0), "test convert graph to ptgraph")
    ("test-load-model", po::value<int>()->default_value(0), "test load model")
    ("test-inference",  po::value<int>()->default_value(0), "test inference")
    ("batch-size",      po::value<int>()->default_value(1), "batch size")
    ("nb-graphs",       po::value<int>()->default_value(1), "nb graphs to load")
    ("data-dir",        po::value<std::string>(),           "graph file path")
    ("graph-file",      po::value<std::string>(),           "graph file path")
    ("model-file",      po::value<std::string>(),           "model file path") ;


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

    if(vm["test-inference"].as<int>() > 0 )
    {
       int opt = vm["test-inference"].as<int>() ;
       std::string model_path = vm["model-file"].as<std::string>();
       std::cout<<"TEST INFERENCE : "<<model_path<<std::endl ;
       torch::jit::script::Module model = read_model(model_path, use_gpu);
       switch(opt)
       {
          case 1 :
          {
            std::vector<PTGraph> graphs ;
            int batch_size = vm["batch-size"].as<int>() ;
            if(vm.count("graph-file"))
            {
              std::string graph_path = vm["graph-file"].as<std::string>();
              std::cout<<"     LOAD DATA : "<<graph_path<<std::endl ;
              Graph graph ;
              loadFromFile(graph,graph_path) ;
              PTGraph pt_graph ;
              convertGraph2PTGraph(graph,pt_graph);
              graphs.push_back(std::move(pt_graph)) ;
            }
            else
            {
              int nb_graphs = vm["nb-graphs"].as<int>() ;
              std::string data_dir = vm["data-dir"].as<std::string>();
              if(batch_size>1)
              {
                 int nb_batch = nb_graphs / batch_size ;
                 int i = 0 ;
                 for(int ib = 0; ib<nb_batch; ++ib)
                 {
                    std::vector<Graph> batch ;
                    for(int j=0;j<batch_size;++j, ++i)
                    {
                         std::stringstream graph_name ;
                         graph_name<<"graph"<<i<<".json";
                         Graph graph ;
                         boost::filesystem::path graph_path(data_dir.c_str());
                         std::string graph_file = (graph_path / graph_name.str()).string() ;
                         std::cout<<"LOAD FILE : "<<graph_file<<std::endl;
                         loadFromFile(graph,graph_file) ;
                         batch.push_back(std::move(graph)) ;
                    }
                    PTGraph pt_graph ;
                    std::cout<<"LOAD BATCH : "<<batch.size()<<std::endl ;
                    convertGraph2PTGraph(batch,pt_graph);
                    //std::cout<<"PTGRAPH : "<<std::endl ;
                    //std::cout<<pt_graph<<std::endl ;
                    graphs.push_back(std::move(pt_graph)) ;
                 }
                 if(i<nb_graphs)
                 {
                    std::vector<Graph> batch ;
                    for( ;i<nb_graphs;++i)
                    {
                         std::stringstream graph_name ;
                         graph_name<<"graph"<<i<<".json";
                         Graph graph ;
                         boost::filesystem::path graph_path(data_dir.c_str());
                         std::string graph_file = (graph_path / graph_name.str()).string() ;
                         std::cout<<"LOAD FILE : "<<graph_file<<std::endl;
                         loadFromFile(graph,graph_file) ;
                         batch.push_back(std::move(graph)) ;
                    }
                    PTGraph pt_graph ;
                    convertGraph2PTGraph(batch,pt_graph);
                    std::cout<<"PTGRAPH : "<<std::endl ;
                    std::cout<<pt_graph<<std::endl ;
                    graphs.push_back(std::move(pt_graph)) ;
                 }
              }
              else
              {
                  for(int i=0;i<nb_graphs;++i)
                  {
                     std::stringstream graph_name ;
                     graph_name<<"graph"<<i<<".json";
                     Graph graph ;
                     boost::filesystem::path graph_path(data_dir.c_str());
                     std::string graph_file = (graph_path / graph_name.str()).string() ;
                     std::cout<<"LOAD FILE : "<<graph_file<<std::endl;
                     loadFromFile(graph,graph_file) ;
                     PTGraph pt_graph ;
                     convertGraph2PTGraph(graph,pt_graph);
                     graphs.push_back(std::move(pt_graph)) ;
                  }
              }
            }
            auto predictions = infer(model, graphs, use_gpu,0,batch_size);
            for(auto& pred : predictions)
            {
                std::cout<<"PRED : "<<std::endl ;
                std::cout<<pred<<std::endl ;
            }
          }
          break ;
          case 2 :
          {
            std::string graph_path = vm["graph-file"].as<std::string>();
            std::cout<<"     LOAD DATA : "<<graph_path<<std::endl ;
            Graph graph ;
            loadFromFile(graph,graph_path) ;
            std::vector<PTGraph> graphs(1) ;
            convertGraph2PTGraph(graph,graphs[0]);
            auto predictions = infer(model, graphs, use_gpu,2,1);
            for(auto& pred : predictions)
            {
                std::cout<<"PRED : "<<std::endl ;
                std::cout<<pred<<std::endl ;
            }
          }
          break ;
          case 3 :
          {
            std::cout<<"    Tensor Data with ones: "<<std::endl ;
            PTGraph pt_graph ;
            assignPTGraphToOnes(pt_graph,10,1) ;
            std::vector<PTGraph> graphs = {pt_graph} ;
            auto predictions = infer(model, graphs, use_gpu,1,1);
            for(auto& pred : predictions)
            {
                std::cout<<"PRED : "<<std::endl ;
                std::cout<<pred<<std::endl ;
            }
          }
          break ;
          case 4 :
          {
            std::cout<<"    Tensor Data with randn: "<<std::endl ;
            PTGraph pt_graph ;
            assignPTGraphToRandn(pt_graph,1,3) ;
            std::vector<PTGraph> graphs = {pt_graph} ;
            auto predictions = infer(model, graphs, use_gpu,1,1);
            for(auto& pred : predictions)
            {
                std::cout<<"PRED : "<<std::endl ;
                std::cout<<pred<<std::endl ;
            }
          }
          break ;

       }

    }


    return 0;
}
