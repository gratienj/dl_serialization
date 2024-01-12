
#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <fstream>
#include <random>
#include <string>
#include <sstream>
#include <memory>

#include "ml4cfd/graph.h"
#include "ml4cfd/internal/graphutils.h"
#include "ml4cfd/internal/modelutils.h"
#include "ml4cfd/internal/cnpy.h"
#include "utils/PerfCounterMng.h"
#include "ml4cfd/DSSSolver.h"


#ifdef USE_TENSORRT
#include "utils/9.2/argsParser.h"
#include "utils/9.2/buffers.h"
#include "utils/9.2/common.h"
#include "utils/9.2/logger.h"
#include "utils/9.2/parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;
#endif

#include "tensorrt/TRTEngineTester.h"


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
    ("test-dss",        po::value<int>()->default_value(0), "test dss")
    ("test-trt",        po::value<int>()->default_value(0), "test trt")
    ("json",            po::value<int>()->default_value(0), "use json format")
    ("normalize",       po::value<int>()->default_value(1), "normalize data")
    ("batch-size",      po::value<int>()->default_value(1), "batch size")
    ("nb-graphs",       po::value<int>()->default_value(1), "nb graphs to load")
    ("niter"    ,       po::value<int>()->default_value(1), "nb iters")
    ("backend",         po::value<std::string>(),           "backend")
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
      ml4cfd::GraphT<float> graph ;
      loadFromFile(graph,graph_path) ;
      //std::cout<<graph<<std::endl ;
    }

    if(vm["test-convert-graph-to-ptgraph"].as<int>() == 1)
    {
      std::string graph_path = vm["graph-file"].as<std::string>();
      std::cout<<"TEST LOAD GRAPH : "<<graph_path<<std::endl ;
      ml4cfd::GraphT<double,int64_t> graph ;
      loadFromFile(graph,graph_path) ;
      ml4cfd::PTGraph pt_graph ;
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
            std::vector<ml4cfd::PTGraphT<float,int64_t>> graphs ;
            int batch_size = vm["batch-size"].as<int>() ;
            if(vm.count("graph-file"))
            {
              std::string graph_path = vm["graph-file"].as<std::string>();
              std::cout<<"     LOAD DATA : "<<graph_path<<std::endl ;
              ml4cfd::GraphT<float,int64_t> graph ;
              loadFromFile(graph,graph_path) ;
              ml4cfd::PTGraphT<float,int64_t> pt_graph ;
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
                    std::vector<ml4cfd::GraphT<float,int64_t>> batch ;
                    for(int j=0;j<batch_size;++j, ++i)
                    {
                         std::stringstream graph_name ;
                         graph_name<<"graph"<<i<<".json";
                         ml4cfd::GraphT<float,int64_t> graph ;
                         boost::filesystem::path graph_path(data_dir.c_str());
                         std::string graph_file = (graph_path / graph_name.str()).string() ;
                         std::cout<<"LOAD FILE : "<<graph_file<<std::endl;
                         loadFromFile(graph,graph_file) ;
                         batch.push_back(std::move(graph)) ;
                    }
                    ml4cfd::PTGraphT<float,int64_t> pt_graph ;
                    std::cout<<"LOAD BATCH : "<<batch.size()<<std::endl ;
                    convertGraph2PTGraph(batch,pt_graph);
                    //std::cout<<"PTGRAPH : "<<std::endl ;
                    //std::cout<<pt_graph<<std::endl ;
                    graphs.push_back(std::move(pt_graph)) ;
                 }
                 if(i<nb_graphs)
                 {
                    std::vector<ml4cfd::GraphT<float,int64_t>> batch ;
                    for( ;i<nb_graphs;++i)
                    {
                         std::stringstream graph_name ;
                         graph_name<<"graph"<<i<<".json";
                         ml4cfd::GraphT<float,int64_t> graph ;
                         boost::filesystem::path graph_path(data_dir.c_str());
                         std::string graph_file = (graph_path / graph_name.str()).string() ;
                         std::cout<<"LOAD FILE : "<<graph_file<<std::endl;
                         loadFromFile(graph,graph_file) ;
                         batch.push_back(std::move(graph)) ;
                    }
                    ml4cfd::PTGraphT<float,int64_t> pt_graph ;
                    convertGraph2PTGraph(batch,pt_graph);
                    std::cout<<"PTGRAPH : "<<std::endl ;
                    //std::cout<<pt_graph<<std::endl ;
                    graphs.push_back(std::move(pt_graph)) ;
                 }
              }
              else
              {
                  for(int i=0;i<nb_graphs;++i)
                  {
                     std::stringstream graph_name ;
                     graph_name<<"graph"<<i<<".json";
                     ml4cfd::GraphT<float,int64_t> graph ;
                     boost::filesystem::path graph_path(data_dir.c_str());
                     std::string graph_file = (graph_path / graph_name.str()).string() ;
                     std::cout<<"LOAD FILE : "<<graph_file<<std::endl;
                     loadFromFile(graph,graph_file) ;
                     ml4cfd::PTGraphT<float,int64_t> pt_graph ;
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
            ml4cfd::GraphT<double,int64_t> graph ;
            loadFromFile(graph,graph_path) ;
            std::vector<ml4cfd::PTGraph> graphs(1) ;
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
            ml4cfd::PTGraph pt_graph ;
            assignPTGraphToOnes(pt_graph,10,1) ;
            std::vector<ml4cfd::PTGraph> graphs = {pt_graph} ;
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
            ml4cfd::PTGraph pt_graph ;
            assignPTGraphToRandn(pt_graph,1,3) ;
            std::vector<ml4cfd::PTGraph> graphs = {pt_graph} ;
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

    if(vm["test-dss"].as<int>() > 0 )
    {
      using namespace ml4cfd ;

      int batch_size = vm["batch-size"].as<int>() ;
      int nb_graphs = vm["nb-graphs"].as<int>() ;
      std::string data_dir = vm["data-dir"].as<std::string>();

      std::string model_path = vm["model-file"].as<std::string>();

      std::cout<<"LOAD DSS MODEL : "<<model_path<<std::endl ;
      DSSSolver solver;
      //solver.init(model_path,DSSSolver::Float32,use_gpu) ;
      solver.initFromConfigFile(model_path) ;

      GraphDataLoader data_loader(solver,batch_size) ;
      GraphResults results ;
      std::vector<std::vector<float>> ys(nb_graphs) ;
      std::vector<ml4cfd::GraphT<float,int64_t>> graphs(nb_graphs) ;
      for(int i=0;i<nb_graphs;++i)
      {
        auto& graph = graphs[i] ;
        if(vm["json"].as<int>()==1)
        {
          //std::string graph_path = vm["graph-file"].as<std::string>();
          std::stringstream case_dir;
          case_dir<<"DDMGraphDSS_0IT_"<<i<<".json";

          boost::filesystem::path root_path(data_dir.c_str());
          std::string graph_path = (root_path / case_dir.str()).string() ;
          std::cout<<"LOAD JSON GRAPH : "<<graph_path<<std::endl ;
          loadFromJsonFile(graph,graph_path,"y") ;
        }
        if(vm["json"].as<int>()==2)
        {
          std::stringstream case_dir;
          case_dir<<"sub_"<<i<<".json";

          boost::filesystem::path root_path(data_dir.c_str());
          boost::filesystem::path dom_path("0");

          std::string graph_path = (root_path / dom_path / case_dir.str()).string() ;
          std::cout<<"LOAD JSON GRAPH : "<<graph_path<<std::endl ;
          loadFromJsonFile(graph,graph_path,"prb_data") ;
        }
        else
        {
          std::stringstream case_dir;
          case_dir<<"CASE_"<<i;
          boost::filesystem::path root_path(data_dir.c_str());
          std::string graph_dir = (root_path / case_dir.str()).string() ;
          std::cout<<"LOAD GRAPH: "<<graph_dir<<std::endl ;
          loadFromFile(graph,graph_dir) ;
        }
        auto graph_id = data_loader.createNewGraph() ;
        data_loader.setGraph(graph_id,
                       graph.m_nb_vertices,
                       graph.m_nb_vertex_attr,
                       graph.m_nb_edges,
                       graph.m_nb_edge_attr,
                       graph.m_y_size,
                       2,
                       graph.m_edge_index.data(),
                       graph.m_x.data(),
                       graph.m_edge_attr.data(),
                       graph.m_y.data(),
                       graph.m_pos.data()) ;
        //data_loader.applyGraphCartesianTransform(graph_id) ;
        if(graph.m_tags.size()>0)
          data_loader.updateGraphVertexTagsData(graph_id,graph.m_tags.data(),graph.m_nb_vertices) ;
        auto y_norm = data_loader.updateGraphPBRData(graph_id,graph.m_y.data(),graph.m_y.size()) ;

        std::cout<<"NORME B : "<<y_norm<<std::endl ;
        ys[i].resize(graph.m_nb_vertices) ;
        ys[i].assign(graph.m_x.begin(),graph.m_x.end()) ;
        results.registerResultsBuffer(graph_id,ys[i].data(),ys[i].size(),ys[i].size(),1.,1.) ;
      }

      for(int it=0;it<vm["niter"].as<int>();++it)
      {
        std::cout<<"NITER : "<<it<<std::endl ;
        for(int i=0;i<nb_graphs;++i)
        {
          auto& graph = graphs[i] ;
          ml4cfd::GraphT<float,int64_t> graph2 ;

          if(vm["json"].as<int>()==1)
          {
            std::stringstream case_dir;
            case_dir<<"DDMGraphDSS_"<<it<<"IT_"<<i<<".json";
            boost::filesystem::path root_path(data_dir.c_str());
            std::string graph_path = (root_path / case_dir.str()).string() ;
            std::cout<<"LOAD JSON GRAPH : "<<graph_path<<std::endl ;
            loadFromJsonFile(graph2,graph_path,"y") ;
          }
          if(vm["json"].as<int>()==2)
          {
            std::stringstream case_dir;
            case_dir<<"sub_"<<i<<".json";

            boost::filesystem::path root_path(data_dir.c_str());
            boost::filesystem::path dom_path("0");

            std::string graph_path = (root_path / dom_path / case_dir.str()).string() ;
            std::cout<<"LOAD JSON GRAPH : "<<graph_path<<std::endl ;
            loadFromJsonFile(graph2,graph_path,"prb_data") ;
          }
          std::cout<<"updateGraphPBRData : "<<graph2.m_y.size()<<std::endl ;
          auto y_norm = data_loader.updateGraphPBRData(i,graph2.m_y.data(),graph2.m_y.size()) ;
        }

        if(vm["normalize"].as<int>()==1)
          data_loader.normalizeData() ;
        solver.solve(data_loader.data(),results) ;
        //results.computePreditionToResults() ;
        float avg_mse = 0. ;
        float max_mse = 0. ;
        for(int i=0;i<nb_graphs;++i)
        {
          auto& graph = graphs[i] ;
          auto& y = ys[i] ;
          float mse = 0. ;
          for(std::size_t j=0;j<y.size();++j)
          {
            float d = y[j] - graph.m_x[j] ;
            mse += d*d ;
            //std::cout<<"SOL["<<i<<"]"<<y[j]<<","<<graph.m_x[j]<<std::endl ;
          }
          mse /= y.size() ;
          avg_mse += mse ;
          max_mse = std::max(max_mse,mse) ;
          std::cout<<"MSE = "<<mse<<std::endl;
        }
        avg_mse /= nb_graphs ;
        std::cout<<"MSE AVG MAX "<<avg_mse<<" "<<max_mse<<std::endl;
      }
    }

    if(vm["test-trt"].as<int>() > 0 )
    {

      std::string model_path = vm["model-file"].as<std::string>();
      std::cout<<"LOAD TRT MODEL : "<<model_path<<std::endl ;

      int batch_size = vm["batch-size"].as<int>() ;

      TRTEngineTester test(model_path,batch_size);


      auto coords0 = cnpy::npy_load("coords0.npy") ;
      auto coords0_dims = coords0.shape.size() ;
      assert(coords0_dims == 2) ;
      std::cout<<"COORDS0 DIMS : "<<coords0.shape<<std::endl ;
      std::vector<float> c0(coords0.data<float>(),coords0.data<float>()+coords0.shape[1]) ;

      auto coords1 = cnpy::npy_load("coords0.npy") ;
      auto coords1_dims = coords1.shape.size() ;
      assert(coords1_dims == 2) ;
      std::cout<<"COORDS1 DIMS : "<<coords1.shape<<std::endl ;
      std::vector<float> c1(coords1.data<float>(),coords1.data<float>()+coords1.shape[1]) ;

      auto edge_to = cnpy::npy_load("edge_to.npy") ;
      auto edge_to_dims = edge_to.shape.size() ;
      assert(edge_to_dims == 2) ;
      std::cout<<"EDGE TO DIMS : "<<edge_to.shape<<std::endl ;
      std::vector<int> e0(edge_to.data<int>(),edge_to.data<int>()+edge_to.shape[1]) ;

      auto edge_from = cnpy::npy_load("edge_from.npy") ;
      auto edge_from_dims = edge_from.shape.size() ;
      assert(edge_from_dims == 2) ;
      std::cout<<"EDGE FROM DIMS : "<<edge_from.shape<<std::endl ;
      std::vector<int> e1(edge_from.data<int>(),edge_from.data<int>()+edge_from.shape[1]) ;

      int nb_vertices = coords0.shape[1] ;
      int nb_edges = edge_to.shape[1] ;
      std::vector<float> output(nb_vertices) ;
      if (!test.build(nb_vertices, nb_edges))
      {
         std::cout<<"TRT UNIT TEST BUILD FAILED";
      }
      if (!test.infer(c0,c1,e0,e1,output,nb_vertices,nb_edges))
      {
         std::cout<<"TRT UNIT TEST INFER FAILED";
      }
    }


    return 0;
}
