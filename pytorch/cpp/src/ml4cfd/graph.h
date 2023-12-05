/*
 * graph.h
 *
 *  Created on: 2 déc. 2023
 *      Author: gratienj
 */


#pragma once
#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <fstream>
#include <random>
#include <string>
#include <memory>


namespace ml4cfd {

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
  std::vector<index_type> m_batch ;
  std::vector<value_type> m_x ;
  std::vector<index_type> m_edge_index ;
  std::vector<value_type> m_edge_attr ;
  std::vector<value_type> m_aij ;
  std::vector<value_type> m_y ;
  std::vector<value_type> m_pos ;
  std::vector<int>        m_tags ;
  value_type              m_dij_max = 1. ;
  value_type              m_L_max = 1. ;
  value_type              m_y_norm = 1. ;
} ;



void loadFromFile(GraphT<double,int64_t>& graph,std::string const& path) ;
void loadFromFile(GraphT<float,int64_t>& graph,std::string const& path) ;

void loadFromJsonFile(GraphT<double,int64_t>& graph,std::string const& path) ;
void loadFromJsonFile(GraphT<float,int64_t>& graph,std::string const& path) ;

template<typename ValueT, typename IndexT>
void loadFromFileT(GraphT<ValueT,IndexT>& graph,std::string const& path) ;

template<typename ValueT, typename IndexT>
void loadFromJsonFileT(GraphT<ValueT,IndexT>& graph,std::string const& path) ;

}

