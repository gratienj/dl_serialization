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
  std::vector<value_type> m_x ;
  std::vector<index_type> m_edge_index ;
  std::vector<value_type> m_edge_attr ;
  std::vector<value_type> m_y ;
} ;

typedef GraphT<> Graph ;

std::ostream& operator <<(std::ostream& ostream,const Graph& graph) ;

void loadFromFile(Graph& graph,std::string const& path) ;


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

