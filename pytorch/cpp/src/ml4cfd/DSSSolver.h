/*
 * DSSSolver.h
 *
 *  Created on: 4 nov. 2023
 *      Author: gratienj
 */

#pragma once

#include <map>
#include <memory>

namespace ml4cfd {

  class GraphData ;

  class GraphResults ;

  class DSSSolver
  {
  public:
    typedef enum {
      Float32,
      Float64
    } ePrecType ;

    DSSSolver();
    virtual ~DSSSolver();

    ePrecType precision() const {
      return m_precision ;
    }

    bool useGpu() const {
      return m_use_gpu ;
    }

    std::size_t batchSize() const {
      return m_batch_size ;
    }

    double getModelFactor() const {
      return m_model_factor ;
    }

    double getDataSetMean() const {
      return m_dataset_mean ;
    }

    double getDataSetStd() const {
      return m_dataset_std ;
    }


    void init(std::string const& model_path,ePrecType prec, bool use_gpu) ;

    void initFromConfigFile(std::string const& config_path) ;

    void end() ;


    bool solve(GraphData const& data, GraphResults& results) ;

  private :
    struct Internal ;
    std::unique_ptr<Internal> m_internal;
    ePrecType m_precision = Float32 ;
    bool m_use_gpu = false ;
    std::size_t m_batch_size = 1 ;
    double m_model_factor = 1. ;
    double m_dataset_mean = 0. ;
    double m_dataset_std = 1. ;

  };

  class GraphDataLoader
  {
  public:
    struct Internal ;
    GraphDataLoader(DSSSolver const& solver, std::size_t batch_size= 0 ) ;
    virtual ~GraphDataLoader() ;
    std::size_t  createNewGraph() ;

    std::size_t graphDataSize() const ;

    std::size_t getGraphNbVertices(std::size_t id) const ;

    double graphNormReal64(std::size_t id) ;

    float graphNormReal32(std::size_t id) ;

    void setGraph(std::size_t id,
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
        double const* pos) ;

    void setGraph(std::size_t id,
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
        float const* pos) ;

    void normalizeData() ;

    void applyGraphCartesianTransform(std::size_t id) ;

    void updateGraphVertexTagsData(std::size_t id, int const* x, std::size_t size) ;

    void updateGraphVertexAttrData(std::size_t id, float const* x, std::size_t size) ;
    void updateGraphVertexAttrData(std::size_t id, double const* x, std::size_t size) ;
    void releasePTGraphVertexAttrData() ;

    float updateGraphPBRData(std::size_t id, float const* y, std::size_t size) ;
    double updateGraphPBRData(std::size_t id, double const* y, std::size_t size) ;
    void releasePTGraphPRBData() ;

    void updateGraphAijData(std::size_t id, float const* aij, std::size_t size) ;
    void updateGraphAijData(std::size_t id, double const* aij, std::size_t size) ;

    void dumpGraphToJsonFile(std::size_t id,std::string const& filename) ;

    void computePTGraphs() ;

    void updatePTGraphData() ;

    GraphData data() ;

  private :

    template<typename GraphT>
    void _setGraph(GraphT& graph,
                   int nb_vertices,
                   int nb_vertex_attr,
                   int nb_edges,
                   int nb_edge_attr,
                   int y_size,
                   int dim,
                   typename GraphT::index_type const* edge_index,
                   typename GraphT::value_type const* vertex_attr,
                   typename GraphT::value_type const* edge_attr,
                   typename GraphT::value_type const* y,
                   typename GraphT::value_type const* pos) ;


    template<typename GraphType, typename PTGraphType>
    void _computePTGraphsT(std::vector<GraphType>& graph_list,
                           std::vector<PTGraphType>& py_graph_list) ;


    template<typename GraphType, typename PTGraphType>
    void _updatePTGraphDataT(std::vector<GraphType>& graph_list,
                              std::vector<PTGraphType>& py_graph_list) ;

    template<typename GraphType>
    void _applyGraphCartesianTransform(GraphType& graph) ;

    template<typename GraphType>
    typename GraphType::value_type _updateGraphPBRDataT(GraphType& graph,
                              typename GraphType::value_type const* y,
                              std::size_t size) ;

    template<typename GraphType>
    void _updateGraphVertexTagsDataT(GraphType& graph,
                                     int const* tags,
                                     std::size_t size) ;

    template<typename GraphType>
    void _updateGraphVertexAttrDataT(GraphType& graph,
                                     typename GraphType::value_type const* x,
                                     std::size_t size) ;

    template<typename GraphType>
    void _updateGraphAijDataT(GraphType& graph,
                              typename GraphType::value_type const* aij,
                              std::size_t size) ;

    template<typename GraphType>
    double _normalizeData(std::vector<GraphType>& graph_list) ;

    DSSSolver const& m_parent;
    DSSSolver::ePrecType m_precision = DSSSolver::Float32 ;
    bool m_use_gpu = false ;
    std::size_t m_batch_size = 1 ;
    double m_dataset_mean = 0. ;
    double m_dataset_std = 1. ;
    double m_normalize_factor = 1. ;
    std::unique_ptr<Internal> m_internal ;
  };

  class GraphData
  {
  public :
    std::size_t m_batch_size = 0 ;
    GraphDataLoader::Internal* m_internal = nullptr ;
    double m_normalize_factor = 1. ;
  };

  template<typename ValueT>
  struct PredictionT
  {
      typedef ValueT value_type ;
      int m_dim0 = 0;
      int m_dim1 = 0;
      std::vector<value_type> m_values ;
  } ;

  template<typename ValueT>
  std::ostream& operator<<(std::ostream& oss,PredictionT<ValueT> const& prediction)
  {
    assert(prediction.m_values.size()>=prediction.m_dim0*prediction.m_dim1) ;
    oss<<"Prediction : dim("<<prediction.m_dim0<<","<<prediction.m_dim1<<")"<<std::endl ;
    oss<<"["<<std::endl ;
    int k = 0 ;
    for(int i=0;i<prediction.m_dim0;++i)
    {
        oss<<"\t[";
        for(int j=0;j<prediction.m_dim1;++j)
           oss<<prediction.m_values[k++]<<" ";
        oss<<"]"<<std::endl ;
    }
    oss<<"]"<<std::endl ;
    return oss;
  }

  class GraphResults
  {
  public:
    template<typename T>
    struct ResultBuffer
    {
      std::size_t m_size = 0 ;
      std::size_t m_restricted_size = 0 ;
      T* m_values = nullptr ;
      T m_norm_factor = 1. ;
    };
    void registerResultsBuffer(std::size_t id,
                               float* buffer,
                               std::size_t size,
                               std::size_t restricted_size,
                               float norm_A,
                               float norm_b)
    {
      m_result32_map[id] = ResultBuffer<float>{size,restricted_size,buffer,norm_b} ;
    }

    void registerResultsBuffer(std::size_t id,
                               double* buffer,
                               std::size_t size,
                               std::size_t restricted_size,
                               double norm_A,
                               double norm_b)
    {
      m_result64_map[id] = ResultBuffer<double>{size,restricted_size,buffer,norm_b} ;
    }

    template<typename value1_type, typename value2_type>
    void computePreditionToResultsT(std::vector<PredictionT<value1_type>> const& prediction,
                                    std::map<std::size_t,ResultBuffer<value2_type>>& results) ;

    void computePreditionToResults() ;

  public:
    DSSSolver::ePrecType m_precision = DSSSolver::Float32 ;
    std::size_t m_batch_size = 0 ;
    std::size_t m_nb_batch = 0 ;
    double m_normalize_factor = 1. ;
    std::vector<PredictionT<double>> m_prediction64_list ;
    std::vector<PredictionT<float>>  m_prediction32_list ;
    std::map<std::size_t,ResultBuffer<float>> m_result32_map ;
    std::map<std::size_t,ResultBuffer<double>> m_result64_map ;

  };

}
