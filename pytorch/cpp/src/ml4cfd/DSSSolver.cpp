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

#include "DSSSolver.h"


#include "internal/graphutils.h"
#include "internal/modelutils.h"



#include <torch/script.h>
#include <torch/serialize/tensor.h>
#include <torch/serialize.h>

namespace ml4cfd {

struct GraphDataLoader::Internal
{
  std::vector<GraphT<float,int64_t>>   m_graph32_list ;
  std::vector<PTGraphT<float,int64_t>> m_pt_graph32_list ;

  std::vector<GraphT<double,int64_t>>   m_graph64_list ;
  std::vector<PTGraphT<double,int64_t>> m_pt_graph64_list ;

  bool m_pt_graph_is_updated = false ;
  bool m_pt_graph_data_is_updated = false ;
} ;


GraphDataLoader::GraphDataLoader(DSSSolver const& parent, int batch_size)
: m_parent(parent)
, m_precision(parent.precision())
, m_use_gpu(parent.useGpu())
, m_batch_size(batch_size)
, m_dataset_mean(parent.getDataSetMean())
, m_dataset_std(parent.getDataSetStd())
{
  m_internal.reset(new GraphDataLoader::Internal) ;
}

GraphDataLoader::~GraphDataLoader()
{}

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
    std::cout<<"SET POS : "<<graph.m_pos.data()<<std::endl ;
    if(pos)
    {
      graph.m_pos.assign( pos, pos+size) ;
      for(int i=0;i<graph.m_nb_vertices;++i)
      {
        std::cout<<"POS["<<i<<"]:("<<graph.m_pos[2*i]<<","<<graph.m_pos[2*i+1]<<")"<<std::endl ;
      }
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
  std::cout<<"COMPUTE GRPHA DATA"<<m_precision<<std::endl ;
  switch(m_precision)
  {
    case DSSSolver::Float32 :
      _computePTGraphsT(m_internal->m_graph32_list,m_internal->m_pt_graph32_list) ;
      break ;
    case DSSSolver::Float64 :
      _computePTGraphsT(m_internal->m_graph64_list,m_internal->m_pt_graph64_list) ;
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
    std::size_t size = std::min(m_batch_size,graph_list.size()-begin) ;
    updateGraph2PTGraphData(graph_list.data()+begin,size,pt_graph_list[ib]);
    begin = begin+size ;
  }
  assert(begin==graph_list.size()) ;
}

void GraphDataLoader::updatePTGraphData()
{
  std::cout<<"UPDATE GRPHA DATA"<<m_precision<<std::endl ;
  switch(m_precision)
  {
    case DSSSolver::Float32 :
      _updatePTGraphDataT(m_internal->m_graph32_list,m_internal->m_pt_graph32_list) ;
      break ;
    case DSSSolver::Float64 :
      _updatePTGraphDataT(m_internal->m_graph64_list,m_internal->m_pt_graph64_list) ;
      break ;
  }
  m_internal->m_pt_graph_data_is_updated = true ;
}


GraphData GraphDataLoader::data()
{
  std::cout<<"GRAPHDATALOADER::DATA ("<<m_internal->m_pt_graph_is_updated<<" "<<m_internal->m_pt_graph_data_is_updated<<")"<<std::endl ;
  if(! m_internal->m_pt_graph_is_updated)
    computePTGraphs() ;

  if(! m_internal->m_pt_graph_data_is_updated)
    updatePTGraphData() ;

  return GraphData{m_batch_size,m_internal.get()} ;
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
      typename GraphType::value_type dij = pos_j[d] - pos_i[d] ;
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
  graph.m_L_max = dij_max ;
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
void GraphDataLoader::_updateGraphVertexAttrDataT(GraphType& graph,
                                                  typename GraphType::value_type const* x,
                                                  std::size_t size)
{
  graph.m_x.assign(x,x+graph.m_nb_vertices*graph.m_nb_vertex_attr) ;
  m_internal->m_pt_graph_data_is_updated = false ;
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
void GraphDataLoader::_updateGraphPBRDataT(GraphType& graph,
                                           typename GraphType::value_type const* y,
                                           std::size_t size)
{
  graph.m_y.assign(y,y+size*graph.m_y_size) ;
  typename GraphType::value_type y_norm = 0 ;
  for(auto y : graph.m_y)
    y_norm += y*y ;
  graph.m_y_norm = std::sqrt(y_norm) ;
  y_norm *= m_dataset_std ;
  for(auto& y : graph.m_y)
    y = (y-m_dataset_mean)/y_norm ;
  m_internal->m_pt_graph_data_is_updated = false ;
}

float GraphDataLoader::updateGraphPBRData(std::size_t id,
                                          float const* y,
                                          std::size_t size)
{
  float y_norm = 1. ;
  switch(m_precision)
    {
      case DSSSolver::Float32 :
      {
        _updateGraphPBRDataT(m_internal->m_graph32_list[id],y,size) ;
        y_norm = m_internal->m_graph32_list[id].m_y_norm ;
      }
      break ;
      case DSSSolver::Float64 :
        std::vector<double> dy(size) ;
        std::copy(y,y+size,dy.data()) ;
        _updateGraphPBRDataT(m_internal->m_graph64_list[id],dy.data(),size) ;
        y_norm = m_internal->m_graph64_list[id].m_y_norm ;
    }
    return y_norm ;
}

double GraphDataLoader::updateGraphPBRData(std::size_t id,
                                           double const* y,
                                           std::size_t size)
{
  double y_norm = 1. ;
  switch(m_precision)
  {
    case DSSSolver::Float32 :
    {
      std::vector<float> fy(size) ;
      std::copy(y,y+size,fy.data()) ;
      std::cout<<"UPDATE PBR["<<id<<"] : "<<size<<std::endl ;
      _updateGraphPBRDataT(m_internal->m_graph32_list[id],fy.data(),size) ;
      y_norm = m_internal->m_graph32_list[id].m_y_norm ;
    }
    break ;
    case DSSSolver::Float64 :
      _updateGraphPBRDataT(m_internal->m_graph64_list[id],y,size) ;
      y_norm = m_internal->m_graph64_list[id].m_y_norm ;
  }
  std::cout<<"NORM B : "<<y_norm<<std::endl ;
  return y_norm ;
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
        std::cout<<"("<<graph.m_pos[i*graph.m_dim]<<","<<graph.m_pos[i*graph.m_dim+1]<<")"<<std::endl ;
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
  value2_type a = 1. ;
  int begin = 0 ;
  std::cout<<"COMPUTE PREDICTIONS TO RESULTS"<<results.size()<<std::endl ;
  for(std::size_t ib = 0; ib<nb_batch; ++ib)
  {
    auto const& pred = prediction_list[ib] ;
    std::cout<<"PRED["<<ib<<"]"<<pred.m_dim0<<" "<<pred.m_dim1<<std::endl ;
    int offset = 0 ;
    int end = std::min(begin+m_batch_size,results.size()) ;
    for(int id = begin;id<end;++id)
    {
      auto&  y = results[id] ;
      std::cout<<"RESULT["<<id<<"]"<<y.m_restricted_size<<" "<<y.m_size<<" "<<y.m_norm_factor<<std::endl ;
      value2_type mse = 0. ;
      value2_type avg = 0. ;
      value2_type avg_ref = 0. ;
      for(int k=0;k<y.m_restricted_size;++k)
      {
        value2_type ref = y.m_values[k] ;
        value2_type val = pred.m_values[offset+k]*y.m_norm_factor ;
        value2_type d = ref - val ;
        avg_ref += ref>0?ref:-ref ;
        avg += val>0?val:-val ;
        mse += d*d ;
        y.m_values[k] = a*val+(1-a)*ref ;
        std::cout<<"   SOL["<<k<<"] "<<y.m_values[k]<<" lu="<<ref<<" pred="<<val<<" factor="<<y.m_norm_factor<<std::endl ;
      }
      std::cout<<"(AVG,AVGREF,MSE,F)=("<<avg<<" "<<avg_ref<<" "<<mse/(begin-end)<<" "<<avg/avg_ref<<")"<<std::endl ;
      offset += y.m_size ;
      std::cout<<"OFFSET : "<<offset<<std::endl;
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
  torch::jit::script::Module m_model ;
} ;

DSSSolver::DSSSolver()
: m_precision(DSSSolver::Float32)
{

}

DSSSolver::~DSSSolver()
{
  //delete m_internal ;
}

void DSSSolver::init(std::string const& model_path,ePrecType prec, bool use_gpu)
{
  m_precision = prec ;
  m_use_gpu = use_gpu ;
  m_internal.reset(new Internal) ;
  m_internal->m_model = read_model(model_path, use_gpu);
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
  bool use_gpu = root.get<int>("gpu",0) > 0 ;

  m_model_factor = root.get<double>("model-factor",1.) ;
  m_dataset_mean = root.get<double>("data-mean",0.) ;
  m_dataset_std = root.get<double>("data-std",1.) ;

  std::string model_path = root.get<std::string>("model");
  m_internal.reset(new Internal) ;
  m_internal->m_model = read_model(model_path, use_gpu);
}

bool DSSSolver::solve(GraphData const& data, GraphResults& results)
{
  switch(m_precision)
  {
    case Float32 :
    {
      results.m_prediction32_list = std::move(infer(m_internal->m_model, data.m_internal->m_pt_graph32_list, m_use_gpu,0,data.m_batch_size));
      results.m_batch_size = data.m_batch_size ;
      for(auto& pred : results.m_prediction32_list)
      {
          std::cout<<"PRED : "<<std::endl ;
          std::cout<<pred<<std::endl ;
      }
      return true ;
    }
    break ;
    case Float64 :
    {
      results.m_batch_size = data.m_batch_size ;
      results.m_prediction64_list = std::move(infer(m_internal->m_model, data.m_internal->m_pt_graph64_list, m_use_gpu,0,data.m_batch_size));
      for(auto& pred : results.m_prediction64_list)
      {
          std::cout<<"PRED : "<<std::endl ;
          std::cout<<pred<<std::endl ;
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
