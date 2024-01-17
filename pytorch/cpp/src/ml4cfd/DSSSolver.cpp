/*
 * DSSSolver.cpp
 *
 *  Created on: 4 nov. 2023
 *      Author: gratienj
 */

#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <fstream>
#include <random>
#include <string>
#include <memory>

//#include <filesystem>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "utils/PerfCounterMng.h"
#include "DSSSolver.h"

#include "graph.h"
#include "internal/graphutils.h"
#include "internal/modelutils.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef USE_TORCH
#include <torch/script.h>
#include <torch/serialize/tensor.h>
#include <torch/serialize.h>
#endif

#ifdef USE_ONNX
//#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#endif

#ifdef USE_TENSORRT
#include "utils/9.2/argsParser.h"
#include "utils/9.2/buffers.h"
#include "utils/9.2/common.h"
#include "utils/9.2/logger.h"
#include "utils/9.2/parserOnnxConfig.h"
#include "NvInfer.h"

using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

#include "internal/TRTEngine.h"
#endif


namespace ml4cfd {
typedef PerfCounterMng<std::string> PerfCounterMngType ;

struct GraphDataLoader::Internal
{
  std::vector<GraphT<float,int64_t>>          m_graph32_list ;
  std::vector<PTGraphT<float,int64_t>>        m_pt_graph32_list ;
  std::vector<ONNXGraphT<float,int64_t>>      m_onnx_graph32_list ;
  std::vector<ONNXGraphT<float,int>>          m_trt_graph32_list ;

  std::vector<GraphT<double,int64_t>>         m_graph64_list ;
  std::vector<PTGraphT<double,int64_t>>       m_pt_graph64_list ;
  std::vector<ONNXGraphT<double,int64_t>>     m_onnx_graph64_list ;
  std::vector<ONNXGraphT<double,int>>         m_trt_graph64_list ;

  bool m_pt_graph_is_updated = false ;
  bool m_pt_graph_data_is_updated = false ;

  mutable PerfCounterMng<std::string> m_perf_mng ;
} ;


GraphDataLoader::GraphDataLoader(DSSSolver const& parent, std::size_t batch_size)
: m_parent(parent)
, m_precision(parent.precision())
, m_backend_rt(parent.backEndRT())
, m_use_gpu(parent.useGpu())
, m_batch_size(parent.batchSize())
, m_dataset_mean(parent.getDataSetMean())
, m_dataset_std(parent.getDataSetStd())
{
  if(batch_size>0)
    m_batch_size = batch_size ;
  m_internal.reset(new GraphDataLoader::Internal) ;
  m_internal->m_perf_mng.init("GraphDataLoader::LoadData") ;
}

GraphDataLoader::~GraphDataLoader()
{
  if(m_internal.get())
  {
    std::cout<<"=============================="<<std::endl ;
    std::cout<<"GRAPH DATA LOADER PERF INFO : "<<std::endl ;
    m_internal->m_perf_mng.printInfo();
    std::cout<<"=============================="<<std::endl ;
  }
}

std::size_t GraphDataLoader::createNewGraph()
{
  switch(m_precision)
  {
    case DSSSolver::Float32:
    {
      std::cout<<"CREATE NEW GRAPH32"<<std::endl ;
      std::size_t id = m_internal->m_graph32_list.size() ;
      GraphT<float,int64_t> graph ;
      m_internal->m_graph32_list.push_back(graph) ;
      return id;
    }
    break ;
    case DSSSolver::Float64:
    {
      std::size_t id = m_internal->m_graph64_list.size() ;
      GraphT<double,int64_t> graph ;
      m_internal->m_graph64_list.push_back(graph) ;
      return id ;
    }
    break ;
    default :
      return -1 ;
  }
}

template<typename GraphT>
void GraphDataLoader::_setGraph(GraphT& graph,
               int nb_vertices,
               int nb_vertex_attr,
               int nb_edges,
               int nb_edge_attr,
               int y_size,
               int dim,
               typename GraphT::index_type const* edge_index,
               typename GraphT::value_type const* x,
               typename GraphT::value_type const* edge_attr,
               typename GraphT::value_type const* y,
               typename GraphT::value_type const* pos)
{
  std::cout<<"SET X : "<<nb_vertices<<" "<<nb_vertex_attr<<std::endl ;
  {
      graph.m_nb_vertices = nb_vertices ;
      graph.m_nb_vertex_attr = nb_vertex_attr;
      std::size_t size = graph.m_nb_vertices*graph.m_nb_vertex_attr ;
      graph.m_x.resize(size) ;
      if(x)
        graph.m_x.assign(x,x+size) ;
  }

  std::cout<<"SET EDGE_INDEX : "<<nb_edges<<" "<<nb_edge_attr<<std::endl ;
  {
      graph.m_nb_edges = nb_edges ;
      graph.m_edge_index.resize(2*graph.m_nb_edges) ;
      graph.m_edge_index.assign(edge_index,edge_index+2*graph.m_nb_edges) ;
  }

  std::cout<<"SET EDGE_ATTR : "<<nb_edges<<" "<<nb_edge_attr<<std::endl ;
  {
    graph.m_nb_edge_attr = nb_edge_attr ;
    std::size_t size = graph.m_nb_edges*graph.m_nb_edge_attr;
    graph.m_edge_attr.resize(size) ;
    if(edge_attr)
      graph.m_edge_attr.assign(edge_attr,edge_attr+size) ;
  }

  std::cout<<"SET Y : "<<nb_vertices<<" "<<y_size<<std::endl ;
  {
    graph.m_y_size = y_size ;
    std::size_t size = graph.m_nb_vertices * y_size ;
    graph.m_y.resize(size) ;
    if(y)
    {
      graph.m_y.assign( y, y+size) ;
      graph.m_y_norm = 0 ;
      for(auto y : graph.m_y)
        graph.m_y_norm += y*y ;
      graph.m_y_norm /= std::sqrt(graph.m_y_norm) ;
      for(auto& y : graph.m_y)
        y /= graph.m_y_norm ;
    }
  }

  std::cout<<"SET POS : "<<nb_vertices<<" "<<dim<<" "<<pos<<std::endl ;
  {
    graph.m_dim = dim ;
    std::size_t size = graph.m_nb_vertices * dim ;
    graph.m_pos.resize(size) ;
    if(pos)
    {
      graph.m_pos.assign( pos, pos+size) ;
      /*
      for(int i=0;i<graph.m_nb_vertices;++i)
      {
        std::cout<<"POS["<<i<<"]:("<<graph.m_pos[2*i]<<","<<graph.m_pos[2*i+1]<<")"<<std::endl ;
      }
      */
    }
  }
}

void GraphDataLoader::setGraph(std::size_t id,
              int nb_vertices,
              int nb_vertex_attr,
              int nb_edges,
              int nb_edges_attr,
              int y_size,
              int dim,
              int64_t const* edge_index,
              double const* vertex_attr,
              double const* edge_attr,
              double const* y,
              double const* pos)
{
  switch(m_precision)
  {
    case DSSSolver::Float64:
    {
      auto& graph = m_internal->m_graph64_list[id] ;
      _setGraph(graph,nb_vertices,nb_vertex_attr,nb_edges,nb_edges_attr,y_size,dim,edge_index,vertex_attr,edge_attr,y,pos) ;
    }
    break ;
    default:
    break ;
  }
}

void GraphDataLoader::setGraph( std::size_t id,
                          int nb_vertices,
                          int nb_vertex_attr,
                          int nb_edges,
                          int nb_edges_attr,
                          int y_size,
                          int dim,
                          int64_t const* edge_index,
                          float const* vertex_attr,
                          float const* edge_attr,
                          float const* y,
                          float const* pos)
{
  switch(m_precision)
  {
    case DSSSolver::Float32:
    {
      std::cout<<"SET GRAPH 32 :"<<id<<std::endl ;
      auto& graph = m_internal->m_graph32_list[id] ;
      _setGraph(graph,nb_vertices,nb_vertex_attr,nb_edges,nb_edges_attr,y_size,dim,edge_index,vertex_attr,edge_attr,y,pos) ;
    }
    break ;
    default:
    break ;
  }
}

std::size_t GraphDataLoader::graphDataSize() const
{
  if(m_internal.get() == nullptr)
    return 0 ;

  switch(m_precision)
    {
      case DSSSolver::Float32:
      {
        return m_internal->m_graph32_list.size() ;
      }
      break ;
      case DSSSolver::Float64:
      {
        return m_internal->m_graph64_list.size() ;
      }
    }
  return 0 ;
}


std::size_t GraphDataLoader::getGraphNbVertices(std::size_t id) const
{
  if(m_internal.get() == nullptr)
        return 1 ;

  switch(m_precision)
  {
    case DSSSolver::Float32:
    {
      return m_internal->m_graph32_list[id].m_nb_vertices ;
    }
    break ;
    case DSSSolver::Float64:
    {
      return m_internal->m_graph64_list[id].m_nb_vertices  ;
    }
  }
  return 0 ;
}

double GraphDataLoader::graphNormReal64(std::size_t id)
{
  if(m_internal.get() == nullptr)
        return 1 ;

  switch(m_precision)
  {
    case DSSSolver::Float32:
    {
      return m_internal->m_graph32_list[id].m_dij_max ;
    }
    break ;
    case DSSSolver::Float64:
    {
      return m_internal->m_graph64_list[id].m_dij_max  ;
    }
  }
  return 1 ;
}

float GraphDataLoader::graphNormReal32(std::size_t id)
{
  if(m_internal.get() == nullptr)
          return 1 ;

  switch(m_precision)
  {
    case DSSSolver::Float32:
    {
      return m_internal->m_graph32_list[id].m_dij_max ;
    }
    break ;
    case DSSSolver::Float64:
    {
      return m_internal->m_graph64_list[id].m_dij_max  ;
    }
  }
  return 1 ;
}

template<typename GraphType, typename PTGraphType>
void GraphDataLoader::_computePTGraphsT(std::vector<GraphType>& graph_list, std::vector<PTGraphType>& pt_graph_list)
{
  int nb_batch = (graph_list.size() + m_batch_size-1) / m_batch_size ;
  std::size_t begin = 0 ;
  for(int ib = 0; ib<nb_batch; ++ib)
  {
    PTGraphType pt_graph ;

    std::size_t size = std::min(m_batch_size,graph_list.size()-begin) ;
    convertGraph2PTGraph(graph_list.data()+begin,size,pt_graph);
    pt_graph_list.push_back(std::move(pt_graph)) ;
    begin = begin+size ;
  }
  assert(begin==graph_list.size()) ;
}

void GraphDataLoader::computePTGraphs()
{
  //std::cout<<"COMPUTE GRAPH DATA"<<m_precision<<std::endl ;
  switch(m_precision)
  {
    case DSSSolver::Float32 :
      switch(m_backend_rt)
      {
        case DSSSolver::Torch:
          m_internal->m_pt_graph32_list.resize(0) ;
          _computePTGraphsT(m_internal->m_graph32_list,m_internal->m_pt_graph32_list) ;
          break ;
        case DSSSolver::ONNX:
          m_internal->m_onnx_graph32_list.resize(0) ;
          _computePTGraphsT(m_internal->m_graph32_list,m_internal->m_onnx_graph32_list) ;
          break ;
        case DSSSolver::TensorRT:
          m_internal->m_trt_graph32_list.resize(0) ;
          _computePTGraphsT(m_internal->m_graph32_list,m_internal->m_trt_graph32_list) ;
          break ;
      }
      break ;
    case DSSSolver::Float64 :
      switch(m_backend_rt)
      {
        case DSSSolver::Torch:
          m_internal->m_pt_graph64_list.resize(0) ;
          _computePTGraphsT(m_internal->m_graph64_list,m_internal->m_pt_graph64_list) ;
          break ;
        case DSSSolver::ONNX:
          m_internal->m_onnx_graph64_list.resize(0) ;
          _computePTGraphsT(m_internal->m_graph64_list,m_internal->m_onnx_graph64_list) ;
          break ;
        case DSSSolver::TensorRT:
          m_internal->m_trt_graph64_list.resize(0) ;
          _computePTGraphsT(m_internal->m_graph64_list,m_internal->m_trt_graph64_list) ;
          break ;
      }
      break ;
  }
  m_internal->m_pt_graph_is_updated = true ;
  m_internal->m_pt_graph_data_is_updated = true ;
}


template<typename GraphType, typename PTGraphType>
void GraphDataLoader::_updatePTGraphDataT(std::vector<GraphType>& graph_list, std::vector<PTGraphType>& pt_graph_list)
{
  int nb_batch = (graph_list.size() + m_batch_size-1) / m_batch_size ;
  std::size_t begin = 0 ;
  for(int ib = 0; ib<nb_batch; ++ib)
  {
    std::cout<<"PTGRAPH["<<ib<<"]"<<pt_graph_list[ib].m_y_is_updated<<std::endl ;
    std::size_t size = std::min(m_batch_size,graph_list.size()-begin) ;
    updateGraph2PTGraphData(graph_list.data()+begin,size,pt_graph_list[ib]);
    begin = begin+size ;
  }
  assert(begin==graph_list.size()) ;
}

void GraphDataLoader::updatePTGraphData()
{
  //std::cout<<"UPDATE GRPHA DATA"<<m_precision<<std::endl ;
  switch(m_precision)
  {
    case DSSSolver::Float32 :
      switch(m_backend_rt)
      {
        case DSSSolver::Torch:
          _updatePTGraphDataT(m_internal->m_graph32_list,m_internal->m_pt_graph32_list) ;
          break ;
        case DSSSolver::ONNX:
          _updatePTGraphDataT(m_internal->m_graph32_list,m_internal->m_onnx_graph32_list) ;
          break ;
        case DSSSolver::TensorRT:
          _updatePTGraphDataT(m_internal->m_graph32_list,m_internal->m_trt_graph32_list) ;
          break ;
      }
      break ;
    case DSSSolver::Float64 :
      switch(m_backend_rt)
      {
        case DSSSolver::Torch:
          _updatePTGraphDataT(m_internal->m_graph64_list,m_internal->m_pt_graph64_list) ;
          break ;
        case DSSSolver::ONNX:
          _updatePTGraphDataT(m_internal->m_graph64_list,m_internal->m_onnx_graph64_list) ;
          break ;
        case DSSSolver::TensorRT:
          _updatePTGraphDataT(m_internal->m_graph64_list,m_internal->m_trt_graph64_list) ;
          break ;
      }
      break ;
  }
  m_internal->m_pt_graph_data_is_updated = true ;
}


template<typename GraphType>
double GraphDataLoader::_normalizeData(std::vector<GraphType>& graph_list)
{
  double norme_y = 0. ;
  for(auto const& g : graph_list)
  {
    for(auto y : g.m_y)
      norme_y += y*y ;
  }

  norme_y = std::sqrt(norme_y) ;
  for(auto& g : graph_list)
  {
    for(std::size_t i=0;i<g.m_y.size();++i)
    {
      auto ref = g.m_y[i] ;
      g.m_y[i] = (g.m_y[i]/norme_y - m_dataset_mean)/ m_dataset_std ;
      //std::cout<<"PRB["<<i<<"]"<<g.m_y[i]<<" "<<ref<<"/"<<norme_y<<"/"<<m_dataset_std<<std::endl ;
    }
  }
  return norme_y ;
}

void GraphDataLoader::normalizeData()
{
  switch(m_precision)
  {
    case DSSSolver::Float32 :
      m_normalize_factor = _normalizeData(m_internal->m_graph32_list) ;
      break ;
    case DSSSolver::Float64 :
      m_normalize_factor = _normalizeData(m_internal->m_graph64_list) ;
      break ;
  }
  std::cout<<"NORMALIZE FACTOR : "<<m_normalize_factor<<std::endl ;
}

GraphData GraphDataLoader::data()
{
  PerfCounterMngType::Sentry sentry(m_internal->m_perf_mng,"GraphDataLoader::LoadData") ;
  std::cout<<"GRAPHDATALOADER::DATA ("<<m_internal->m_pt_graph_is_updated<<" "<<m_internal->m_pt_graph_data_is_updated<<")"<<std::endl ;
  if(! m_internal->m_pt_graph_is_updated)
    computePTGraphs() ;

  if(! m_internal->m_pt_graph_data_is_updated)
    updatePTGraphData() ;

  return GraphData{m_batch_size,m_internal.get(),m_normalize_factor} ;
}



template<typename GraphType>
void GraphDataLoader::_applyGraphCartesianTransform(GraphType& graph)
{
  std::cout<<"APPLY CARTESIAN TRANSFORM : "<<graph.m_nb_edges<<" "<<graph.m_dim<<" "<<graph.m_pos.data()<<std::endl ;
  typename GraphType::value_type L_max = 0;
  typename GraphType::value_type dij_max = 0;
  for(int e=0;e<graph.m_nb_edges;++e)
  {
    int vi = graph.m_edge_index[e] ;
    int vj = graph.m_edge_index[graph.m_nb_edges+e] ;
    //std::cout<<"EDGE("<<vi<<","<<vj<<") POS : ";
    typename GraphType::value_type* pos_i = graph.m_pos.data() + graph.m_dim*vi ;
    typename GraphType::value_type* pos_j = graph.m_pos.data() + graph.m_dim*vj ;
    typename GraphType::value_type distance = 0. ;
    for(int d=0;d<graph.m_dim;++d)
    {
      typename GraphType::value_type dij = pos_i[d] - pos_j[d] ;
      //std::cout<<dij<<",";
      dij_max = std::max(dij_max,(dij>0?dij:-dij)) ;
      graph.m_edge_attr[e*graph.m_nb_edge_attr+d] = dij ;
      distance += dij*dij ;
    }
    //std::cout<<")"<<std::endl ;
    distance = std::sqrt(distance) ;
    L_max = std::max(L_max,distance) ;
    graph.m_edge_attr[e*graph.m_nb_edge_attr+graph.m_dim] = distance ;
  }
  dij_max *= 2 ;
  graph.m_dij_max = dij_max ;
  graph.m_L_max = L_max ;
  std::cout<<"NORMALISATION (LMAX,DMAX):("<<L_max<<","<<dij_max<<std::endl ;
  for(int e=0;e<graph.m_nb_edges;++e)
  {
    for(int d=0;d<graph.m_dim;++d)
    {
      typename GraphType::value_type dij = graph.m_edge_attr[e*graph.m_nb_edge_attr+d] ;
      graph.m_edge_attr[e*graph.m_nb_edge_attr+d] = 0.5 + dij/dij_max ;
    }
    graph.m_edge_attr[e*graph.m_nb_edge_attr+graph.m_dim] /= graph.m_L_max ;
  }
}

void GraphDataLoader::applyGraphCartesianTransform(std::size_t id)
{
  std::cout<<"APPLY TRANSFORM "<<id<<" "<<m_internal->m_graph32_list[id].m_pos.data()<<std::endl ;
  switch(m_precision)
  {
    case DSSSolver::Float32 :
      _applyGraphCartesianTransform(m_internal->m_graph32_list[id]) ;
      break ;
    case DSSSolver::Float64 :
      _applyGraphCartesianTransform(m_internal->m_graph64_list[id]) ;
      break ;
  }
}

template<typename GraphType>
void GraphDataLoader::_updateGraphVertexTagsDataT(GraphType& graph,
                                                  int const* tags,
                                                  std::size_t size)
{
  graph.m_tags.resize(graph.m_nb_vertices) ;
  std::copy(tags,tags+size,graph.m_tags.data()) ;
}

template<typename GraphType>
void GraphDataLoader::_updateGraphVertexAttrDataT(GraphType& graph,
                                                  typename GraphType::value_type const* x,
                                                  std::size_t size)
{
  std::copy(x,x+size*graph.m_nb_vertex_attr,graph.m_x.data()) ;
  m_internal->m_pt_graph_data_is_updated = false ;
}

void GraphDataLoader::updateGraphVertexTagsData(std::size_t id,
                                                int const* tags,
                                                std::size_t size)
{
  switch(m_precision)
  {
    case DSSSolver::Float32 :
    {
      _updateGraphVertexTagsDataT(m_internal->m_graph32_list[id],tags,size) ;
    }
    break ;
    case DSSSolver::Float64 :
      _updateGraphVertexTagsDataT(m_internal->m_graph64_list[id],tags,size) ;
      break ;
  }
}

void GraphDataLoader::updateGraphVertexAttrData(std::size_t id,
                                                float const* x,
                                                std::size_t size)
{
  _updateGraphVertexAttrDataT(m_internal->m_graph32_list[id],x,size) ;
}

void GraphDataLoader::updateGraphVertexAttrData(std::size_t id,
                                                double const* x,
                                                std::size_t size)
{
  switch(m_precision)
  {
    case DSSSolver::Float32 :
    {
      std::vector<float> fx(size) ;
      std::copy(x,x+size,fx.data()) ;
      std::cout<<"UPDATE VERTEX ATTR["<<id<<"] : "<<size<<std::endl ;
      _updateGraphVertexAttrDataT(m_internal->m_graph32_list[id],fx.data(),size) ;
    }
    break ;
    case DSSSolver::Float64 :
      _updateGraphVertexAttrDataT(m_internal->m_graph64_list[id],x,size) ;
      break ;
  }
}

template<typename GraphType>
typename GraphType::value_type GraphDataLoader::_updateGraphPBRDataT(GraphType& graph,
                                           typename GraphType::value_type const* y,
                                           std::size_t size)
{
  std::copy(y,y+size*graph.m_y_size,graph.m_y.data()) ;
  typename GraphType::value_type y_norm = 0 ;
  for(std::size_t i=0;i<size;++i)
  {
    y_norm += y[i]*y[i] ;
  }
  graph.m_y_norm = std::sqrt(y_norm) ;
  //for(auto& y : graph.m_y)
  //  y = (y/graph.m_y_norm-m_dataset_mean)/m_dataset_std ;
  /*
  for(std::size_t i=0;i<size;++i)
  {
    graph.m_y[i] = (graph.m_y[i]-m_dataset_mean)/m_dataset_std ;
  }*/
  m_internal->m_pt_graph_data_is_updated = false ;
  //std::cout<<"NORM B : "<<graph.m_y_norm<<" "<<m_dataset_std<<std::endl ;
  /*
  for(std::size_t i=0;i<size;++i)
  {
    std::cout<<"PRB["<<i<<"]"<<graph.m_y[i]<<std::endl ;
  }*/
  return  graph.m_y_norm ;
}

float GraphDataLoader::updateGraphPBRData(std::size_t id,
                                          float const* y,
                                          std::size_t size)
{
  switch(m_precision)
  {
    case DSSSolver::Float32 :
    {
      return _updateGraphPBRDataT(m_internal->m_graph32_list[id],y,size) ;
    }
    break ;
    case DSSSolver::Float64 :
      std::vector<double> dy(size) ;
      std::copy(y,y+size,dy.data()) ;
      return _updateGraphPBRDataT(m_internal->m_graph64_list[id],dy.data(),size) ;
  }
  return 1. ;
}

double GraphDataLoader::updateGraphPBRData(std::size_t id,
                                           double const* y,
                                           std::size_t size)
{
  switch(m_precision)
  {
    case DSSSolver::Float32 :
    {
      std::vector<float> fy(size) ;
      std::copy(y,y+size,fy.data()) ;
      //std::cout<<"UPDATE PBR["<<id<<"] : "<<size<<std::endl ;
      return _updateGraphPBRDataT(m_internal->m_graph32_list[id],fy.data(),size) ;
    }
    break ;
    case DSSSolver::Float64 :
      return _updateGraphPBRDataT(m_internal->m_graph64_list[id],y,size) ;
  }
  return 1.;
}

void GraphDataLoader::releasePTGraphVertexAttrData()
{
  switch(m_precision)
  {
    case DSSSolver::Float32 :
    {
      for(auto& g : m_internal->m_pt_graph32_list)
        g.m_x_is_updated = false ;
    }
    break ;
    case DSSSolver::Float64 :
    {
      for(auto& g : m_internal->m_pt_graph64_list)
        g.m_x_is_updated = false ;
    }
  }
}

void GraphDataLoader::releasePTGraphPRBData()
{
  switch(m_precision)
  {
    case DSSSolver::Float32 :
    {
      std::cout<<"RELEASE PT GRAPH x32 PRB : "<<m_internal->m_pt_graph32_list.size()<<std::endl ;;

      for(auto& g : m_internal->m_pt_graph32_list)
        g.m_y_is_updated = false ;
    }
    break ;
    case DSSSolver::Float64 :
    {
      std::cout<<"RELEASE PT GRAPH x64 PRB : "<<m_internal->m_pt_graph32_list.size() ;
      for(auto& g : m_internal->m_pt_graph64_list)
        g.m_y_is_updated = false ;

    }
  }

}

template<typename GraphType>
void GraphDataLoader::_updateGraphAijDataT(GraphType& graph,
                                           typename GraphType::value_type const* aij,
                                           std::size_t size)
{
  graph.m_aij.resize(graph.m_nb_edges) ;
  std::copy(aij,aij+size,graph.m_aij.data()) ;
}

void GraphDataLoader::updateGraphAijData(std::size_t id,
                                         float const* aij,
                                         std::size_t size)
{
  switch(m_precision)
  {
    case DSSSolver::Float32 :
    {
      _updateGraphAijDataT(m_internal->m_graph32_list[id],aij,size) ;
    }
    break ;
    case DSSSolver::Float64 :
      std::vector<double> daij(size) ;
      std::copy(aij,aij+size,daij.data()) ;
      _updateGraphAijDataT(m_internal->m_graph64_list[id],daij.data(),size) ;
  }
}

void GraphDataLoader::updateGraphAijData(std::size_t id,
                                         double const* aij,
                                         std::size_t size)
{
  switch(m_precision)
  {
    case DSSSolver::Float32 :
    {
      std::vector<float> faij(size) ;
      std::copy(aij,aij+size,faij.data()) ;
      std::cout<<"UPDATE AIJ["<<id<<"] : "<<size<<std::endl ;
      _updateGraphAijDataT(m_internal->m_graph32_list[id],faij.data(),size) ;
    }
    break ;
    case DSSSolver::Float64 :
      _updateGraphAijDataT(m_internal->m_graph64_list[id],aij,size) ;
  }
}

template<typename value_type, typename index_type>
void dumpToJsonFileT(GraphT<value_type,index_type> const& graph,std::string const& file_path)
{
  namespace pt = boost::property_tree;
  pt::ptree root;
  {
    pt::ptree x_node;
    for (int i = 0; i < graph.m_nb_vertices; i++)
    {
        pt::ptree row;
        for (int j = 0; j < graph.m_nb_vertex_attr; j++)
        {
            // Create an unnamed value
            pt::ptree elem;
            elem.put_value(graph.m_x[i*graph.m_nb_vertex_attr+j]);
            // Add the value to our row
            row.push_back(std::make_pair("", elem));
        }
        // Add the row to our matrix
        x_node.push_back(std::make_pair("", row));
    }
    root.add_child("x", x_node);
  }
  {
    pt::ptree edge_index_node;
    int offset = 0 ;
    for (int i = 0; i < 2; i++)
    {
        pt::ptree row;
        for (int j = 0; j < graph.m_nb_edges; j++)
        {
            // Create an unnamed value
            pt::ptree elem;
            elem.put_value(graph.m_edge_index[offset+j]);
            // Add the value to our row
            row.push_back(std::make_pair("", elem));
        }
        // Add the row to our matrix
        edge_index_node.push_back(std::make_pair("", row));
        offset += graph.m_nb_edges ;
    }
    root.add_child("edge_index", edge_index_node);
  }

  {
    pt::ptree edge_attr_node;
    for (int i = 0; i < graph.m_nb_edges; i++)
    {
        pt::ptree row;
        for (int j = 0; j < graph.m_nb_edge_attr; j++)
        {
            // Create an unnamed value
            pt::ptree elem;
            elem.put_value(graph.m_edge_attr[i*graph.m_nb_edge_attr+j]);
            // Add the value to our row
            row.push_back(std::make_pair("", elem));
        }
        // Add the row to our matrix
        edge_attr_node.push_back(std::make_pair("", row));
    }
    root.add_child("edge_attr", edge_attr_node);
  }

  {
    pt::ptree y_node;
    for (int i = 0; i < graph.m_nb_vertices; i++)
    {
        pt::ptree row;
        for (int j = 0; j < graph.m_y_size; j++)
        {
            // Create an unnamed value
            pt::ptree elem;
            elem.put_value(graph.m_y[i*graph.m_y_size+j]);
            // Add the value to our row
            row.push_back(std::make_pair("", elem));
        }
        // Add the row to our matrix
        y_node.push_back(std::make_pair("", row));
    }
    root.add_child("y", y_node);
  }

  {
    pt::ptree pos_node;
    std::cout<<"DUMP POS : dim="<<graph.m_dim<<std::endl ;
    for (int i = 0; i < graph.m_nb_vertices; i++)
    {
        pt::ptree row;
        //std::cout<<"("<<graph.m_pos[i*graph.m_dim]<<","<<graph.m_pos[i*graph.m_dim+1]<<")"<<std::endl ;
        for (int j = 0; j < graph.m_dim; j++)
        {
            // Create an unnamed value
            pt::ptree elem;
            elem.put_value(graph.m_pos[i*graph.m_dim+j]);
            // Add the value to our row
            row.push_back(std::make_pair("", elem));
        }
        // Add the row to our matrix
        pos_node.push_back(std::make_pair("", row));
    }
    root.add_child("pos", pos_node);
  }

  if(graph.m_tags.size()>0)
  {
    pt::ptree tags_node;
    //std::cout<<"DUMP TAGS "<<std::endl ;
    for (int i = 0; i < graph.m_nb_vertices; i++)
    {
        pt::ptree row;
        //std::cout<<"("<<i<<","<<graph.m_tags[i]<<")"<<std::endl ;
        // Create an unnamed value
        pt::ptree elem;
        elem.put_value(graph.m_tags[i]);
        // Add the value to our row
        row.push_back(std::make_pair("", elem));
        // Add the row to our matrix
        tags_node.push_back(std::make_pair("", row));
    }
    root.add_child("tags", tags_node);
  }


  if(graph.m_aij.size()>0)
  {
    pt::ptree aij_node;
    //std::cout<<"DUMP AIJ "<<std::endl ;
    for (int i = 0; i < graph.m_nb_edges; i++)
    {
        pt::ptree row;
        //std::cout<<"("<<i<<","<<graph.m_aij[i]<<")"<<std::endl ;
        // Create an unnamed value
        pt::ptree elem;
        elem.put_value(graph.m_aij[i]);
        // Add the value to our row
        row.push_back(std::make_pair("", elem));
        // Add the row to our matrix
        aij_node.push_back(std::make_pair("", row));
    }
    root.add_child("aij", aij_node);
  }

  std::ofstream fout(file_path) ;

  pt::write_json(fout, root);
}

void GraphDataLoader::dumpGraphToJsonFile(std::size_t id, std::string const& filename)
{
  std::stringstream file;
  file<<filename<<"_"<<id<<".json";
  switch(m_precision)
  {
    case DSSSolver::Float32 :
    {
      dumpToJsonFileT<float,int64_t>(m_internal->m_graph32_list[id],file.str()) ;
    }
    break ;
    case DSSSolver::Float64 :
    {
      dumpToJsonFileT<double,int64_t>(m_internal->m_graph64_list[id],file.str()) ;
    }
  }
}

template<typename value1_type, typename value2_type>
void GraphResults::
computePreditionToResultsT(std::vector<PredictionT<value1_type>> const& prediction_list,
                           std::map<std::size_t,ResultBuffer<value2_type>>& results)
{
  std::size_t nb_batch = prediction_list.size() ;
  int begin = 0 ;
  //std::cout<<"COMPUTE PREDICTIONS TO RESULTS "<<results.size()<<std::endl ;
  for(std::size_t ib = 0; ib<nb_batch; ++ib)
  {
    auto const& pred = prediction_list[ib] ;
    //std::cout<<"PRED["<<ib<<"]"<<pred.m_dim0<<" "<<pred.m_dim1<<std::endl ;
    int offset = 0 ;
    int end = std::min(begin+m_batch_size,results.size()) ;
    for(int id = begin;id<end;++id)
    {
      auto&  y = results[id] ;
#ifdef DEBUG
      std::cout<<"RESULT["<<id<<"]"<<y.m_restricted_size<<" "<<y.m_size<<" "<<y.m_norm_factor<<" "<<m_normalize_factor<<std::endl ;
      value2_type mse = 0. ;
#endif
      for(int k=0;k<y.m_restricted_size;++k)
      {
        value2_type val = pred.m_values[offset+k]*m_normalize_factor ;
#ifdef DEBUG
        value2_type ref = y.m_values[k] ;
        value2_type d = ref - val ;
        mse += d*d ;
#endif
        y.m_values[k] = val ;
        //std::cout<<"   SOL["<<k<<"]="<<y.m_values[k]<<","<<ref<<","<<pred.m_values[offset+k]<<" factor="<<m_normalize_factor<<std::endl ;
        //std::cout<<"   SOL["<<k<<"]="<<y.m_values[k]<<","<<ref<<std::endl ;
      }
#ifdef DEBUG
      if(y.m_restricted_size>0)
        mse /= y.m_restricted_size ;
      std::cout<<"MSE="<<mse<<" "<<mse/m_normalize_factor<<std::endl ;
#endif
      offset += y.m_size ;
      //std::cout<<"OFFSET : "<<offset<<std::endl;
    }
    assert(offset == pred.m_dim0) ;
    begin = end ;
  }
}

void GraphResults::computePreditionToResults()
{
  if(m_prediction32_list.size()>0)
  {
    switch(m_precision)
      {
        case DSSSolver::Float32 :
        {
          computePreditionToResultsT(m_prediction32_list,m_result32_map) ;
        }
        break ;
        case DSSSolver::Float64 :
        {
          computePreditionToResultsT(m_prediction32_list,m_result64_map) ;
        }
      break ;
      }
  }
  if( m_prediction64_list.size()>0)
  {
    switch(m_precision)
      {
        case DSSSolver::Float32 :
        {
          computePreditionToResultsT(m_prediction64_list,m_result32_map) ;
        }
        break ;
        case DSSSolver::Float64 :
        {
          computePreditionToResultsT(m_prediction64_list,m_result64_map) ;
        }
      break ;
      }
  }
}

struct DSSSolver::Internal
{
  mutable PerfCounterMng<std::string> m_perf_mng ;
} ;

struct DSSSolver::TorchInternal
{
  torch::jit::script::Module m_model ;
  int m_torch_count = 0 ;
  mutable PerfCounterMng<std::string> m_perf_mng ;
} ;


struct DSSSolver::ONNXInternal
{
  ONNXInternal()
#ifdef USE_ONNX
: m_environment(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING)
#endif
  {}

#ifdef USE_ONNX
    Ort::Env m_environment;
    std::unique_ptr<Ort::Session> m_session;
    cudaStream_t m_cuda_compute_stream = nullptr ;
    OrtCUDAProviderOptionsV2* m_cuda_options = nullptr;
#endif
    mutable PerfCounterMng<std::string> m_perf_mng ;
} ;


struct DSSSolver::TensorRTInternal
{
  std::unique_ptr<TRTEngine> m_trt_engine ;
  mutable PerfCounterMng<std::string> m_perf_mng ;
} ;

DSSSolver::DSSSolver()
{
  std::cout<<"DSSSOLVER DEFAULT CONSTRUCTOR : "<<std::endl ;
}

DSSSolver::~DSSSolver()
{
  std::cout<<"DSSSOLVER PERF INFO : "<<std::endl ;
  if(m_torch_internal.get())
    m_torch_internal->m_perf_mng.printInfo();
  if(m_onnx_internal.get())
  {
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if(m_onnx_internal->m_cuda_options)
      api->ReleaseCUDAProviderOptions(m_onnx_internal->m_cuda_options);

    if(m_onnx_internal->m_cuda_compute_stream)
      cudaStreamDestroy(m_onnx_internal->m_cuda_compute_stream);

    std::cout<<"======================"<<std::endl ;
    m_onnx_internal->m_perf_mng.printInfo();
  }
  if(m_tensorrt_internal.get())
    m_tensorrt_internal->m_perf_mng.printInfo();
  if(m_internal.get())
    m_internal->m_perf_mng.printInfo();
  std::cout<<"======================"<<std::endl ;
}

void DSSSolver::init(std::string const& model_path, ePrecType prec, eBackEndRT backend_rt, bool use_gpu)
{
  m_precision = prec ;
  m_backend_rt = backend_rt ;
  m_use_gpu = use_gpu ;
  m_internal.reset(new Internal) ;
  m_internal->m_perf_mng.init("DSS::ComputeResults") ;
  m_internal->m_perf_mng.init("DSS::Solve") ;
  switch(m_backend_rt)
  {
    case Torch :
      {
        m_torch_internal.reset(new TorchInternal) ;
        m_torch_internal->m_perf_mng.init("Torch::Init") ;
        m_torch_internal->m_perf_mng.init("Torch::Prepare") ;
        m_torch_internal->m_perf_mng.init("Torch::Compute0") ;
        m_torch_internal->m_perf_mng.init("Torch::Compute") ;
        m_torch_internal->m_perf_mng.init("Torch::End") ;
        PerfCounterMngType::Sentry sentry(m_torch_internal->m_perf_mng,"Torch::Prepare") ;
        m_torch_internal->m_model = read_model(model_path, use_gpu);
      }
      break ;
    case ONNX :
      {
        m_internal->m_perf_mng.init("DSS::CreateTensor") ;
        m_internal->m_perf_mng.init("DSS::ONNXRun") ;
        m_onnx_internal.reset(new ONNXInternal) ;
        m_onnx_internal->m_perf_mng.init("ONNX::Init") ;
        m_onnx_internal->m_perf_mng.init("ONNX::Prepare") ;
        m_onnx_internal->m_perf_mng.init("ONNX::Compute") ;
        m_onnx_internal->m_perf_mng.init("ONNX::End") ;
        PerfCounterMngType::Sentry sentry(m_onnx_internal->m_perf_mng,"ONNX::Prepare") ;
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        m_onnx_internal->m_session = std::make_unique<Ort::Session>(m_onnx_internal->m_environment, model_path.c_str(), session_options);
      }
      break ;
    case TensorRT :
      {
        m_tensorrt_internal.reset(new TensorRTInternal) ;
        m_tensorrt_internal->m_perf_mng.init("TensorRT::Init") ;
        m_tensorrt_internal->m_perf_mng.init("TensorRT::Prepare") ;
        m_tensorrt_internal->m_perf_mng.init("TensorRT::Compute") ;
        m_tensorrt_internal->m_perf_mng.init("TensorRT::End") ;
        PerfCounterMngType::Sentry sentry(m_tensorrt_internal->m_perf_mng,"TensorRT::Prepare") ;
        m_tensorrt_internal->m_trt_engine.reset( new TRTEngine(model_path,m_batch_size));
      }
      break ;
  }
}

void DSSSolver::initFromConfigFile(std::string const& config_path)
{

  namespace pt = boost::property_tree;
  // Create a root
  pt::ptree root;

  // Load the json file in this ptree
  pt::read_json(config_path, root);

  int precision = root.get<int>("precision",32);
  if(precision==32)
    m_precision = Float32 ;
  if(precision==64)
    m_precision = Float64 ;
  m_use_gpu = root.get<int>("gpu",0) > 0 ;

  m_batch_size = root.get<int>("batch-size",1) ;

  m_model_factor = root.get<double>("model-factor",1.) ;
  m_dataset_mean = root.get<double>("dataset-mean",0.) ;
  m_dataset_std = root.get<double>("dataset-std",1.) ;

  std::string backend = root.get<std::string>("backend","torch");
  std::string model_path = root.get<std::string>("model");
  m_internal.reset(new Internal) ;
  m_internal->m_perf_mng.init("DSS::ComputeResults") ;
  m_internal->m_perf_mng.init("DSS::Solve") ;
  if(backend.compare("torch")==0)
  {
    m_backend_rt = Torch ;
#ifdef USE_TORCH
    m_torch_internal.reset(new TorchInternal) ;
    m_torch_internal->m_perf_mng.init("Torch::Init") ;
    m_torch_internal->m_perf_mng.init("Torch::Prepare") ;
    m_torch_internal->m_perf_mng.init("Torch::Compute0") ;
    m_torch_internal->m_perf_mng.init("Torch::Compute") ;
    m_torch_internal->m_perf_mng.init("Torch::End") ;
    m_torch_internal->m_torch_count = 0 ;
    PerfCounterMngType::Sentry sentry(m_torch_internal->m_perf_mng,"Torch::Prepare") ;
    m_torch_internal->m_model = read_model(model_path, m_use_gpu);
#endif
  }
  if(backend.compare("onnx")==0)
  {
    m_backend_rt = ONNX ;
#ifdef USE_ONNX
    m_onnx_internal.reset(new ONNXInternal) ;
    m_onnx_internal->m_perf_mng.init("ONNX::Init") ;
    m_onnx_internal->m_perf_mng.init("ONNX::Prepare") ;
    m_onnx_internal->m_perf_mng.init("ONNX::Compute") ;
    m_onnx_internal->m_perf_mng.init("ONNX::End") ;
    PerfCounterMngType::Sentry sentry(m_onnx_internal->m_perf_mng,"ONNX::Prepare") ;
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    if(m_use_gpu)
    {
      cudaStreamCreateWithFlags(&m_onnx_internal->m_cuda_compute_stream, cudaStreamNonBlocking);
      //MyCustomOp custom_op{onnxruntime::kCudaExecutionProvider};
      const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
      api->CreateCUDAProviderOptions(&m_onnx_internal->m_cuda_options);

      std::vector<const char*> keys{"device_id",
                                    "gpu_mem_limit",
                                    "arena_extend_strategy",
                                    "cudnn_conv_algo_search",
                                    "do_copy_in_default_stream",
                                    "cudnn_conv_use_max_workspace",
                                    "cudnn_conv1d_pad_to_nc1d"};
      std::vector<const char*> values{"0",
                                      "2147483648",
                                      "kSameAsRequested",
                                      "DEFAULT",
                                      "1",
                                      "1",
                                      "1"};

      api->UpdateCUDAProviderOptions(m_onnx_internal->m_cuda_options, keys.data(), values.data(), keys.size());
      api->UpdateCUDAProviderOptionsWithValue(m_onnx_internal->m_cuda_options, "user_compute_stream", m_onnx_internal->m_cuda_compute_stream) ;

      std::cout << "Running simple inference with cuda provider" << std::endl;
      //auto cuda_options = CreateDefaultOrtCudaProviderOptionsWithCustomStream(m_onnx_internal->m_cuda_compute_stream);
      session_options.AppendExecutionProvider_CUDA_V2(*m_onnx_internal->m_cuda_options);
    }

    m_onnx_internal->m_session = std::make_unique<Ort::Session>(m_onnx_internal->m_environment, model_path.c_str(), session_options);
#endif
  }
  if(backend.compare("tensorrt")==0)
  {
    m_backend_rt = TensorRT ;
#ifdef USE_TENSORRT

    m_tensorrt_internal.reset(new TensorRTInternal) ;
    m_tensorrt_internal->m_perf_mng.init("TensorRT::Init") ;
    m_tensorrt_internal->m_perf_mng.init("TensorRT::Prepare") ;
    m_tensorrt_internal->m_perf_mng.init("TensorRT::Compute") ;
    m_tensorrt_internal->m_perf_mng.init("TensorRT::End") ;
    PerfCounterMngType::Sentry sentry(m_tensorrt_internal->m_perf_mng,"TensorRT::Prepare") ;
    m_tensorrt_internal->m_trt_engine.reset( new TRTEngine(model_path,m_batch_size));
#endif
  }
}


void DSSSolver::initPerfInfo()
{
  if(m_internal.get())
  {
    m_internal->m_perf_mng.init("DSS::ComputeResults") ;
    m_internal->m_perf_mng.init("DSS::Solve") ;
  }

  switch(m_backend_rt)
  {
    case Torch:
      if(m_torch_internal.get())
      {
        m_torch_internal->m_perf_mng.init("Torch::Init") ;
        m_torch_internal->m_perf_mng.init("Torch::Prepare") ;
        m_torch_internal->m_perf_mng.init("Torch::Compute0") ;
        m_torch_internal->m_perf_mng.init("Torch::Compute") ;
        m_torch_internal->m_perf_mng.init("Torch::End") ;
      }
      break ;
    case ONNX :
      if(m_onnx_internal.get())
      {
        m_onnx_internal->m_perf_mng.init("ONNX::Init") ;
        m_onnx_internal->m_perf_mng.init("ONNX::Prepare") ;
        m_onnx_internal->m_perf_mng.init("ONNX::Compute") ;
        m_onnx_internal->m_perf_mng.init("ONNX::End") ;
      }
      break ;
    case TensorRT:
      if(m_tensorrt_internal.get())
      {
        m_tensorrt_internal->m_perf_mng.init("TensorRT::Init") ;
        m_tensorrt_internal->m_perf_mng.init("TensorRT::Prepare") ;
        m_tensorrt_internal->m_perf_mng.init("TensorRT::Compute") ;
        m_tensorrt_internal->m_perf_mng.init("TensorRT::End") ;
      }
  }
}
template<typename value_type, typename PTGraphT>
std::vector<ml4cfd::PredictionT<value_type> >
forwardT(std::vector<PTGraphT>& graphs,
        Ort::Session* session,
        bool usegpu,
        PerfCounterMngType& perf_mng)
{
  std::vector<ml4cfd::PredictionT<value_type>> predictions(graphs.size()) ;
  if (usegpu)
  {
    std::cout<<"FORWARD ON GPU : NB GRAPHS : "<<graphs.size()<<std::endl ;
    std::size_t icount = 0 ;
    for(auto& g : graphs)
    {
      // onnx tensors:
      Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

      // input:
      std::vector<Ort::Value> input_tensors;
      {
        PerfCounterMngType::Sentry sentry(perf_mng,"DSS::CreateTensor") ;
        Ort::TypeInfo input_type_info = session->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_dims = input_tensor_info.GetShape();
        assert(input_dims.size()==2) ;
        assert(input_dims[0]==-1) ;
        assert(input_dims[1]==1) ;
        // batch size
        input_dims[0] = g.m_total_nb_vertices;
        std::cout<<"NB VERTICES : "<<g.m_total_nb_vertices<<std::endl ;
        input_tensors.push_back(Ort::Value::CreateTensor<value_type>(memory_info,
                                                                const_cast<value_type*>(g.m_x.data()),
                                                                g.m_x.size(),
                                                                input_dims.data(),
                                                                input_dims.size()));
      }
      {
        PerfCounterMngType::Sentry sentry(perf_mng,"DSS::CreateTensor") ;
        Ort::TypeInfo input_type_info = session->GetInputTypeInfo(1);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_dims = input_tensor_info.GetShape();
        assert(input_dims.size()==2) ;
        assert(input_dims[0]==2) ;
        assert(input_dims[1]==-1) ;
        input_dims[1]= g.m_total_nb_edges ;
        /*
        std::cout<<"NB EDGES : "<<g.m_total_nb_edges<<std::endl ;
        std::cout<<"EDGE INDEX : "<<std::endl ;
        for(int e=0;e<g.m_total_nb_edges;++e)
        {
          std::cout<<"EDGE["<<e<<"] : ["<<g.m_edge_index[e]<<","<<g.m_edge_index[e+g.m_total_nb_edges]<<"]"<<std::endl ;
        }*/
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info,
                                                                  const_cast<int64_t*>(g.m_edge_index.data()),
                                                                  g.m_edge_index.size(),
                                                                  input_dims.data(),
                                                                  input_dims.size()));
      }
      {
        PerfCounterMngType::Sentry sentry(perf_mng,"DSS::CreateTensor") ;
        Ort::TypeInfo input_type_info = session->GetInputTypeInfo(2);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_dims = input_tensor_info.GetShape();
        assert(input_dims.size()==2) ;
        assert(input_dims[0]==-1) ;
        assert(input_dims[1]==3) ;
        assert(g.m_edge_attr.size() == g.m_total_nb_edges*3) ;
        input_dims[0] = g.m_total_nb_edges;
        input_tensors.push_back(Ort::Value::CreateTensor<value_type>(memory_info,
                                                                  const_cast<value_type*>(g.m_edge_attr.data()),
                                                                  g.m_edge_attr.size(),
                                                                  input_dims.data(),
                                                                  input_dims.size()));
      }
      {
        PerfCounterMngType::Sentry sentry(perf_mng,"DSS::CreateTensor") ;
        Ort::TypeInfo input_type_info = session->GetInputTypeInfo(3);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_dims = input_tensor_info.GetShape();
        assert(input_dims.size()==2) ;
        assert(input_dims[0]==-1) ;
        assert(input_dims[1]==1) ;
        assert(g.m_y.size() == g.m_total_nb_vertices) ;
        input_dims[0] = g.m_total_nb_vertices;
        input_tensors.push_back(Ort::Value::CreateTensor<value_type>(memory_info,
                                                                  const_cast<value_type*>(g.m_y.data()),
                                                                  g.m_y.size(),
                                                                  input_dims.data(),
                                                                  input_dims.size()));
      }


      // output:
      std::vector<Ort::Value> output_tensors;
      {
        PerfCounterMngType::Sentry sentry(perf_mng,"DSS::CreateTensor") ;
        Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_dims = output_tensor_info.GetShape();
        assert(output_dims.size()==2) ;
        assert(output_dims[0]==-1) ;
        assert(output_dims[1]==1) ;
        output_dims[0] = g.m_total_nb_vertices ;

        auto& prediction = predictions[icount++] ;
        prediction.m_dim0 = g.m_total_nb_vertices ;
        prediction.m_dim1 = 1 ;

        prediction.m_values.resize(prediction.m_dim0 * prediction.m_dim1);
        output_tensors.push_back(Ort::Value::CreateTensor<value_type>(memory_info,
                                                                 prediction.m_values.data(),
                                                                 prediction.m_values.size(),
                                                                 output_dims.data(), output_dims.size()));
      }


      // serving names:
      std::string input0_name = session->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get();
      std::string input1_name = session->GetInputNameAllocated(1, Ort::AllocatorWithDefaultOptions()).get();
      std::string input2_name = session->GetInputNameAllocated(2, Ort::AllocatorWithDefaultOptions()).get();
      std::string input3_name = session->GetInputNameAllocated(3, Ort::AllocatorWithDefaultOptions()).get();

      std::vector<const char*>  input_names{input0_name.c_str(),
                                            input1_name.c_str(),
                                            input2_name.c_str(),
                                            input3_name.c_str()};
      std::string output_name = session->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get();
      std::vector<const char*> output_names{output_name.c_str()};

      std::cout<<" ONNX MODEL INFERENCE ON GPU"<<std::endl ;
      PerfCounterMngType::Sentry sentry(perf_mng,"DSS::ONNXRun") ;
     // model inference
      session->Run(Ort::RunOptions{nullptr},
                   input_names.data(),
                   input_tensors.data(), 4,
                   output_names.data(),
                   output_tensors.data(), 1);

      std::cout<<"onnx inference ok"<<std::endl;

    }

  }
  else
  {
      std::cout<<"FORWARD ON CPU : NB GRAPHS : "<<graphs.size()<<std::endl ;
      std::size_t icount = 0 ;
      for(auto& g : graphs)
      {
        // onnx tensors:
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        // input:
        std::vector<Ort::Value> input_tensors;
        {
          PerfCounterMngType::Sentry sentry(perf_mng,"DSS::CreateTensor") ;
          Ort::TypeInfo input_type_info = session->GetInputTypeInfo(0);
          auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
          std::vector<int64_t> input_dims = input_tensor_info.GetShape();
          assert(input_dims.size()==2) ;
          assert(input_dims[0]==-1) ;
          assert(input_dims[1]==1) ;
          // batch size
          input_dims[0] = g.m_total_nb_vertices;
          std::cout<<"NB VERTICES : "<<g.m_total_nb_vertices<<std::endl ;
          input_tensors.push_back(Ort::Value::CreateTensor<value_type>(memory_info,
                                                                  const_cast<value_type*>(g.m_x.data()),
                                                                  g.m_x.size(),
                                                                  input_dims.data(),
                                                                  input_dims.size()));
        }
        {
          PerfCounterMngType::Sentry sentry(perf_mng,"DSS::CreateTensor") ;
          Ort::TypeInfo input_type_info = session->GetInputTypeInfo(1);
          auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
          std::vector<int64_t> input_dims = input_tensor_info.GetShape();
          assert(input_dims.size()==2) ;
          assert(input_dims[0]==2) ;
          assert(input_dims[1]==-1) ;
          input_dims[1]= g.m_total_nb_edges ;
          /*
          std::cout<<"NB EDGES : "<<g.m_total_nb_edges<<std::endl ;
          std::cout<<"EDGE INDEX : "<<std::endl ;
          for(int e=0;e<g.m_total_nb_edges;++e)
          {
            std::cout<<"EDGE["<<e<<"] : ["<<g.m_edge_index[e]<<","<<g.m_edge_index[e+g.m_total_nb_edges]<<"]"<<std::endl ;
          }*/
          input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(memory_info,
                                                                    const_cast<int64_t*>(g.m_edge_index.data()),
                                                                    g.m_edge_index.size(),
                                                                    input_dims.data(),
                                                                    input_dims.size()));
        }
        {
          PerfCounterMngType::Sentry sentry(perf_mng,"DSS::CreateTensor") ;
          Ort::TypeInfo input_type_info = session->GetInputTypeInfo(2);
          auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
          std::vector<int64_t> input_dims = input_tensor_info.GetShape();
          assert(input_dims.size()==2) ;
          assert(input_dims[0]==-1) ;
          assert(input_dims[1]==3) ;
          assert(g.m_edge_attr.size() == g.m_total_nb_edges*3) ;
          input_dims[0] = g.m_total_nb_edges;
          input_tensors.push_back(Ort::Value::CreateTensor<value_type>(memory_info,
                                                                    const_cast<value_type*>(g.m_edge_attr.data()),
                                                                    g.m_edge_attr.size(),
                                                                    input_dims.data(),
                                                                    input_dims.size()));
        }
        {
          PerfCounterMngType::Sentry sentry(perf_mng,"DSS::CreateTensor") ;
          Ort::TypeInfo input_type_info = session->GetInputTypeInfo(3);
          auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
          std::vector<int64_t> input_dims = input_tensor_info.GetShape();
          assert(input_dims.size()==2) ;
          assert(input_dims[0]==-1) ;
          assert(input_dims[1]==1) ;
          assert(g.m_y.size() == g.m_total_nb_vertices) ;
          input_dims[0] = g.m_total_nb_vertices;
          input_tensors.push_back(Ort::Value::CreateTensor<value_type>(memory_info,
                                                                    const_cast<value_type*>(g.m_y.data()),
                                                                    g.m_y.size(),
                                                                    input_dims.data(),
                                                                    input_dims.size()));
        }


        // output:
        std::vector<Ort::Value> output_tensors;
        {
          PerfCounterMngType::Sentry sentry(perf_mng,"DSS::CreateTensor") ;
          Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(0);
          auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
          std::vector<int64_t> output_dims = output_tensor_info.GetShape();
          assert(output_dims.size()==2) ;
          assert(output_dims[0]==-1) ;
          assert(output_dims[1]==1) ;
          output_dims[0] = g.m_total_nb_vertices ;

          auto& prediction = predictions[icount++] ;
          prediction.m_dim0 = g.m_total_nb_vertices ;
          prediction.m_dim1 = 1 ;

          prediction.m_values.resize(prediction.m_dim0 * prediction.m_dim1);
          output_tensors.push_back(Ort::Value::CreateTensor<value_type>(memory_info,
                                                                   prediction.m_values.data(),
                                                                   prediction.m_values.size(),
                                                                   output_dims.data(), output_dims.size()));
        }


        // serving names:
        std::string input0_name = session->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get();
        std::string input1_name = session->GetInputNameAllocated(1, Ort::AllocatorWithDefaultOptions()).get();
        std::string input2_name = session->GetInputNameAllocated(2, Ort::AllocatorWithDefaultOptions()).get();
        std::string input3_name = session->GetInputNameAllocated(3, Ort::AllocatorWithDefaultOptions()).get();

        std::vector<const char*>  input_names{input0_name.c_str(),
                                              input1_name.c_str(),
                                              input2_name.c_str(),
                                              input3_name.c_str()};
        std::string output_name = session->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get();
        std::vector<const char*> output_names{output_name.c_str()};

        std::cout<<" ONNX MODEL INFERENCE"<<std::endl ;
       // model inference
        PerfCounterMngType::Sentry sentry(perf_mng,"DSS::ONNXRun") ;
        session->Run(Ort::RunOptions{nullptr},
                     input_names.data(),
                     input_tensors.data(), 4,
                     output_names.data(),
                     output_tensors.data(), 1);

        std::cout<<"onnx inference ok"<<std::endl;

      }
  }
  return predictions;
}

template<typename value_type, typename PTGraphT>
std::vector<ml4cfd::PredictionT<value_type> >
forwardRT(std::vector<PTGraphT>& graphs,
          TRTEngine* trt_engine,
          bool usegpu)
{
  std::cout<<"FORWARD : NB GRAPHS : "<<graphs.size()<<std::endl ;
  std::vector<ml4cfd::PredictionT<value_type>> predictions(graphs.size()) ;

  if (usegpu)
  {
    std::cerr<<"NOT YET IMPLEMENTED"<<std::endl ;
  }
  else
  {
      std::size_t icount = 0 ;
      for(auto& g : graphs)
      {
        bool ok = trt_engine->build(g.m_total_nb_vertices,g.m_total_nb_edges) ;
        if(!ok)
        {
           std::cerr<<"ENGINE BUILD FAILED"<<std::endl ;;
        }
        else
        {
          auto& prediction = predictions[icount++] ;
          prediction.m_dim0 = g.m_total_nb_vertices ;
          prediction.m_dim1 = 1 ;
          prediction.m_values.resize(prediction.m_dim0 * prediction.m_dim1);
          trt_engine->infer(g.m_x,
                            g.m_edge_index,
                            g.m_edge_attr,
                            g.m_y,
                            prediction.m_values,
                            g.m_total_nb_vertices,
                            g.m_total_nb_edges) ;

          std::cout<<"tensorrt inference ok"<<std::endl;
        }
      }
  }
  return predictions;
}



std::vector<ml4cfd::PredictionT<float> >
DSSSolver::_infer32(GraphData const& data, bool use_gpu, int nb_args, std::size_t batch_size)
{
  switch(m_backend_rt)
  {
    case Torch:
    {
      if(m_torch_internal->m_torch_count<5)
      {
        std::cout<<"FIRST TORCH COMPUTE "<<m_torch_internal->m_torch_count<<std::endl ;
        ++m_torch_internal->m_torch_count ;
        PerfCounterMngType::Sentry sentry(m_torch_internal->m_perf_mng,"Torch::Compute0") ;
        return infer(m_torch_internal->m_model, data.m_internal->m_pt_graph32_list, use_gpu,nb_args, batch_size) ;
      }
      else
      {
        std::cout<<"OTHER TORCH COMPUTE "<<m_torch_internal->m_torch_count<<std::endl ;
        ++m_torch_internal->m_torch_count ;
        PerfCounterMngType::Sentry sentry(m_torch_internal->m_perf_mng,"Torch::Compute") ;
        return infer(m_torch_internal->m_model, data.m_internal->m_pt_graph32_list, use_gpu,nb_args, batch_size) ;

      }
    }
    case ONNX:
    {
      PerfCounterMngType::Sentry sentry(m_onnx_internal->m_perf_mng,"ONNX::Compute") ;
      return forwardT<float,ONNXGraphT<float,int64_t>>(data.m_internal->m_onnx_graph32_list,
                                                       m_onnx_internal->m_session.get(),
                                                       use_gpu,
                                                       m_internal->m_perf_mng) ;
    }
    case TensorRT:
    {
      PerfCounterMngType::Sentry sentry(m_tensorrt_internal->m_perf_mng,"TensorRT::Compute") ;
      return forwardRT<float,ONNXGraphT<float,int>>(data.m_internal->m_trt_graph32_list,m_tensorrt_internal->m_trt_engine.get(),use_gpu) ;
    }
    default:
      return std::vector<ml4cfd::PredictionT<float> >() ;
  }
}

std::vector<ml4cfd::PredictionT<double> >
DSSSolver::_infer64(GraphData const& data, bool use_gpu, int nb_args, std::size_t batch_size)
{
  switch(m_backend_rt)
  {
    case Torch:
      return infer(m_torch_internal->m_model, data.m_internal->m_pt_graph64_list, use_gpu,nb_args, batch_size) ;
    case ONNX:
      return forwardT<double,ONNXGraphT<double,int64_t>>(data.m_internal->m_onnx_graph64_list,
                                                         m_onnx_internal->m_session.get(),
                                                         use_gpu,
                                                         m_internal->m_perf_mng) ;
    case TensorRT:
      return forwardRT<double,ONNXGraphT<double,int>>(data.m_internal->m_trt_graph64_list,m_tensorrt_internal->m_trt_engine.get(),use_gpu) ;
    default:
      return std::vector<ml4cfd::PredictionT<double> >();
  }
}


bool DSSSolver::solve(GraphData const& data, GraphResults& results)
{
  PerfCounterMngType::Sentry sentry(m_internal->m_perf_mng,"DSS::Solve") ;
  switch(m_precision)
  {
    case Float32 :
    {
      results.m_normalize_factor = data.m_normalize_factor ;
      results.m_batch_size = data.m_batch_size ;
      auto prediction32_list = std::move(_infer32(data,m_use_gpu,0, data.m_batch_size));
      /*
      for(auto& pred : results.m_prediction32_list)
      {
          std::cout<<"PRED : "<<std::endl ;
          std::cout<<pred<<std::endl ;
      }
      */
      switch(results.m_precision)
      {
        case Float32 :
        {
          PerfCounterMngType::Sentry sentry(m_internal->m_perf_mng,"DSS::ComputeResults") ;
          results.computePreditionToResultsT(prediction32_list,results.m_result32_map) ;
        }
        break ;
        case Float64 :
        {
          PerfCounterMngType::Sentry sentry(m_internal->m_perf_mng,"DSS::ComputeResults") ;
          results.computePreditionToResultsT(prediction32_list,results.m_result64_map) ;
        }
        break ;
      }
      return true ;
    }
    break ;
    case Float64 :
    {
      results.m_normalize_factor = data.m_normalize_factor ;
      results.m_batch_size = data.m_batch_size ;
      auto prediction64_list = std::move(_infer64(data,m_use_gpu,0,data.m_batch_size));
       /*
      for(auto& pred : results.m_prediction64_list)
      {
          std::cout<<"PRED : "<<std::endl ;
          std::cout<<pred<<std::endl ;
      }*/
      switch(results.m_precision)
      {
        case Float32 :
        {
          PerfCounterMngType::Sentry sentry(m_internal->m_perf_mng,"DSS::ComputeResults") ;
          results.computePreditionToResultsT(prediction64_list,results.m_result32_map) ;
        }
        break ;
        case Float64 :
        {
          PerfCounterMngType::Sentry sentry(m_internal->m_perf_mng,"DSS::ComputeResults") ;
          results.computePreditionToResultsT(prediction64_list,results.m_result64_map) ;
        }
        break ;
      }
      return true ;

    }
    break ;
    default :
    break ;
  }
  return false ;
}

}
