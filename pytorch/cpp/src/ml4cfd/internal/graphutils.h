#pragma once
#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <fstream>
#include <random>
#include <string>
#include <memory>

#include <torch/script.h>
#include <torch/serialize/tensor.h>
#include <torch/serialize.h>

namespace ml4cfd {

typedef GraphT<double,int64_t> Graph ;

std::ostream& operator <<(std::ostream& ostream,const GraphT<double,int64_t>& graph) ;

template<typename ValueT=double, typename IndexT=int64_t>
class PTGraphT
{
public :
  typedef ValueT value_type ;
  typedef IndexT index_type ;
  torch::Tensor m_batch ;
  torch::Tensor m_x ;
  torch::Tensor m_edge_index ;
  torch::Tensor m_edge_attr ;
  torch::Tensor m_y ;
  bool m_x_is_updated          = false ;
  bool m_edge_index_is_updated = false ;
  bool m_edge_attr_is_updated  = false ;
  bool m_y_is_updated          = false ;

  torch::Device m_batch_device = torch::Device(torch::kCPU);
  torch::Device m_x_device = torch::Device(torch::kCPU);
  torch::Device m_edge_index_device = torch::Device(torch::kCPU);
  torch::Device m_edge_attr_device = torch::Device(torch::kCPU);
  torch::Device m_y_device = torch::Device(torch::kCPU);

  void to(torch::Device device) {
    if(m_x_device!=device)
    {
      m_x = m_x.to(device) ;
      m_x_device = device ;
    }
    if(m_edge_index_device!=device)
    {
      m_edge_index = m_edge_index.to(device) ;
      m_edge_index_device = device ;
    }
    if(m_edge_attr_device!=device)
    {
      m_edge_attr = m_edge_attr.to(device) ;
      m_edge_attr_device = device ;
    }
    if(m_y_device!=device)
    {
      m_y = m_y.to(device) ;
      m_y_device = device ;
    }
  }
} ;

typedef PTGraphT<> PTGraph ;

void loadFromPTFile(PTGraph& graph,std::string const& path) ;

std::ostream& operator <<(std::ostream& ostream,const PTGraph& graph) ;

void convertGraph2PTGraph(Graph& graph, PTGraph& pt_graph) ;

void convertGraph2PTGraph(GraphT<float,int64_t>& graph, PTGraphT<float,int64_t>& pt_graph) ;

void convertGraph2PTGraph(std::vector<Graph>& batch, PTGraph& pt_graph) ;
void convertGraph2PTGraph(Graph* begin,int batch_size, PTGraph& pt_graph) ;
void updateGraph2PTGraphData(Graph* begin,int batch_size, PTGraph& pt_graph) ;

void convertGraph2PTGraph(std::vector<GraphT<float,int64_t>>& batch, PTGraphT<float,int64_t>& pt_graph) ;
void convertGraph2PTGraph(GraphT<float,int64_t>* begin, int batch_size, PTGraphT<float,int64_t>& pt_graph) ;
void updateGraph2PTGraphData(GraphT<float,int64_t>* begin, int batch_size, PTGraphT<float,int64_t>& pt_graph) ;

void assignPTGraphToOnes(PTGraph& pt_graph, int64_t dim0, int64_t dim1) ;

void assignPTGraphToRandn(PTGraph& pt_graph, int64_t dim0, int64_t dim1) ;
}
