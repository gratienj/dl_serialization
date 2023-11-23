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

template<typename ValueT=double, typename IndexT=int64_t>
class GraphT
{
public :
  typedef ValueT value_type ;
  typedef IndexT index_type ;
  std::size_t             m_nb_vertices = 0;
  std::size_t             m_nb_edges    = 0;
  std::size_t             m_nb_vertex_attr = 0 ;
  std::size_t             m_nb_edge_attr   = 0 ;
  std::size_t             m_y_size         = 0 ;
  int                     m_dim            = 2 ;
  std::vector<index_type>    m_batch ;
  std::vector<value_type> m_x ;
  std::vector<index_type> m_edge_index ;
  std::vector<value_type> m_edge_attr ;
  std::vector<value_type> m_y ;
  std::vector<value_type> m_pos ;
  value_type              m_dij_max = 1. ;
  value_type              m_L_max = 1. ;
  value_type              m_y_norm = 1. ;
} ;

typedef GraphT<> Graph ;

std::ostream& operator <<(std::ostream& ostream,const Graph& graph) ;

void loadFromFile(Graph& graph,std::string const& path) ;
void loadFromFile(GraphT<float,int64_t>& graph,std::string const& path) ;

void loadFromJsonFile(Graph& graph,std::string const& path) ;
void loadFromJsonFile(GraphT<float,int64_t>& graph,std::string const& path) ;

template<typename ValueT, typename IndexT>
void loadFromFileT(GraphT<ValueT,IndexT>& graph,std::string const& path) ;

template<typename ValueT, typename IndexT>
void loadFromJsonFileT(GraphT<ValueT,IndexT>& graph,std::string const& path) ;

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
