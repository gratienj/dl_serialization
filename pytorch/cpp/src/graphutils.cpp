#include "graphutils.h"
//#include <filesystem>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

void loadFromPTFile(PTGraph& graph,std::string const& dir_path)
{
  //std::filesystem::path graph_path(dir_path.c_str());
  boost::filesystem::path graph_path(dir_path.c_str());
  std::cout<<"LOAD FILE : "<<(graph_path / "X.pt").string()<<std::endl;
  torch::load(graph.m_x,(graph_path / "X.pt").string()) ;

  std::cout<<"LOAD FILE : "<<(graph_path / "E.pt").string()<<std::endl;
  torch::load(graph.m_edge_index,(graph_path / "E.pt").string()) ;

  std::cout<<"LOAD FILE : "<<(graph_path / "Y.pt").string()<<std::endl;
  torch::load(graph.m_y,(graph_path / "Y.pt").string()) ;
}

void loadFromFile(Graph& graph,std::string const& file_path)
{
    namespace pt = boost::property_tree;

    // Create a root
    pt::ptree root;

    // Load the json file in this ptree
    pt::read_json(file_path, root);
    {
        graph.m_nb_vertices = 0 ;
        int count = 0 ;
        //std::cout<<"LOAD X"<<std::endl ;
        for (pt::ptree::value_type &x : root.get_child("x"))
        {
            for (pt::ptree::value_type &d : x.second)
            {
               graph.m_x.push_back(d.second.get_value<Graph::value_type>()) ;
               ++count ;
            }
            ++graph.m_nb_vertices ;
        }
        graph.m_nb_vertex_attr = count / graph.m_nb_vertices ;
        //std::cout<<"NB VERTICES : "<<graph.m_nb_vertices<<std::endl;
        //std::cout<<"NB VERTEX ATTR : "<<graph.m_nb_vertex_attr<<std::endl;
    }
    {
        //std::cout<<"LOAD E"<<std::endl ;
        int count = 0 ;
        for (pt::ptree::value_type &adj : root.get_child("edge_index"))
        {
            for (pt::ptree::value_type &d : adj.second)
            {
                graph.m_edge_index.push_back(d.second.get_value<Graph::index_type>());
               ++count ;
            }
        }
        graph.m_nb_edges = count / 2 ;
    }
    {
        //std::cout<<"LOAD EDGE ATTR"<<std::endl ;
        int count = 0 ;
        for (pt::ptree::value_type &attr : root.get_child("edge_attr"))
        {
            for (pt::ptree::value_type &d : attr.second)
            {
                graph.m_edge_attr.push_back(d.second.get_value<Graph::value_type>());
               ++count ;
            }
        }
        graph.m_nb_edge_attr = count / graph.m_nb_edges ;
    }
    {
        //std::cout<<"LOAD Y"<<std::endl ;
        for (pt::ptree::value_type &y : root.get_child("y"))
        {
            for (pt::ptree::value_type &d : y.second)
            {
               graph.m_y.push_back(d.second.get_value<Graph::value_type>()) ;
            }
        }
        graph.m_y_size = graph.m_y.size() ;
    }
}

std::ostream& operator <<(std::ostream& ostream,const Graph& graph)
{
  ostream<<" GRAPH INFO : "<<std::endl ;
  ostream<<" NB VERTICES : "<<graph.m_nb_vertices<<std::endl ;
  ostream<<" NB VERTEX ATTR : "<<graph.m_nb_vertex_attr<<std::endl ;
  ostream<<" NB EDGES : "<<graph.m_nb_edges<<std::endl ;
  ostream<<" NB EDGE ATTR : "<<graph.m_nb_edge_attr<<std::endl ;
  ostream<<" NB LABELS : "<<graph.m_y_size<<std::endl ;
  if(graph.m_batch.size()>0)
  {
    ostream<<"BATCH:"<<std::endl ;
    ostream<<"["<<std::endl ;
    for(int v=0;v<graph.m_nb_vertices;++v)
      ostream<<graph.m_batch[v]<<" ";
    ostream<<"]"<<std::endl ;
  }
  ostream<<"X:"<<std::endl ;
  {
      int icount = 0 ;
      ostream<<"["<<std::endl ;
      for(int v=0;v<graph.m_nb_vertices;++v)
      {
        ostream<<"\t[";
        for(int i=0;i<graph.m_nb_vertex_attr;++i)
        {
          ostream<<graph.m_x[icount++]<<" ";
        }
        ostream<<"]"<<std::endl ;
      }
      ostream<<"]"<<std::endl ;
  }
  ostream<<"EDGE:"<<std::endl ;
  {
      int icount = 0 ;
      ostream<<"["<<std::endl ;
      for(int e=0;e<2;++e)
      {
        ostream<<"\t[";
        for(int i=0;i<graph.m_nb_edges;++i)
        {
          ostream<<graph.m_edge_index[icount++]<<" ";
        }
        ostream<<"]"<<std::endl ;
      }
      ostream<<"]"<<std::endl ;
  }

  ostream<<"EDGE ATTR:"<<std::endl ;
  {
      int icount = 0 ;
      ostream<<"["<<std::endl ;
      for(int e=0;e<graph.m_nb_edges;++e)
      {
        ostream<<"\t[";
        for(int i=0;i<graph.m_nb_edge_attr;++i)
        {
          ostream<<graph.m_edge_attr[icount++]<<" ";
        }
        ostream<<"]"<<std::endl ;
      }
      ostream<<"]"<<std::endl ;
  }
  ostream<<"Y:"<<std::endl ;
  {
      ostream<<"[" ;
      for(int i=0;i<graph.m_y.size();++i)
      {
          ostream<<graph.m_y[i]<<" ";
      }
      ostream<<"]"<<std::endl ;
  }
  return ostream ;
}


std::ostream& operator <<(std::ostream& ostream,const PTGraph& graph)
{
  ostream<<" PTGRAPH INFO : "<<std::endl ;
  ostream<<"X:"<<std::endl ;
  ostream<<graph.m_x<<std::endl ;
  ostream<<"EDGE:"<<std::endl ;
  ostream<<graph.m_edge_index<<std::endl ;
  ostream<<"EDGE ATTR:"<<std::endl ;
  ostream<<graph.m_edge_attr<<std::endl ;
  ostream<<"Y:"<<std::endl ;
  ostream<<graph.m_y<<std::endl ;
}

void convertGraph2PTGraph(Graph& graph, PTGraph& pt_graph)
{
   {
       torch::TensorOptions options(torch::kInt64);
       pt_graph.m_batch = torch::zeros(graph.m_nb_vertices,options).clone() ;
   }
   {
       std::vector<int64_t> dims = {graph.m_nb_vertices, graph.m_nb_vertex_attr};
       torch::TensorOptions options(torch::kFloat64);
       pt_graph.m_x = torch::from_blob(graph.m_x.data(), torch::IntList(dims), options).clone();
   }
   {
       std::vector<int64_t> dims = {2, graph.m_nb_edges};
       torch::TensorOptions options(torch::kInt64);
       pt_graph.m_edge_index = torch::from_blob(graph.m_edge_index.data(), torch::IntList(dims), options).clone();
   }
   {
       std::vector<int64_t> dims = {graph.m_nb_edges,graph.m_nb_edge_attr};
       torch::TensorOptions options(torch::kFloat64);
       pt_graph.m_edge_attr = torch::from_blob(graph.m_edge_attr.data(), torch::IntList(dims), options).clone();
   }
   {
       std::vector<int64_t> dims = {1, graph.m_y_size};
       torch::TensorOptions options(torch::kFloat64);
       pt_graph.m_y = torch::from_blob(graph.m_y.data(), torch::IntList(dims), options).clone();
   }
}


void convertGraph2PTGraph(std::vector<Graph>& graph_batch, PTGraph& pt_graph)
{
   std::cout<<"CONVERT BATCH TO PTGRAPH : "<<graph_batch.size()<<std::endl ;
   std::vector<int64_t>           batch;
   std::vector<Graph::value_type> x ;
   std::vector<int64_t>           edge_index;
   std::vector<Graph::value_type> edge_attr ;
   std::vector<Graph::value_type> y ;
   std::size_t batch_size        = graph_batch.size() ;
   std::size_t nb_total_vertices = 0 ;
   std::size_t nb_total_edges    = 0 ;
   std::size_t nb_vertex_attr    = graph_batch[0].m_nb_vertex_attr ;
   std::size_t nb_edge_attr      = graph_batch[0].m_nb_edge_attr ;
   std::size_t y_size            = graph_batch[0].m_y_size ;
   for(auto const& g : graph_batch)
   {
       nb_total_vertices += g.m_nb_vertices ;
       nb_total_edges += g.m_nb_edges ;
       assert(g.m_nb_vertex_attr == nb_vertex_attr) ;
       assert(g.m_nb_edge_attr == nb_edge_attr) ;
   }
   batch.resize(nb_total_vertices) ;
   x.reserve(nb_total_vertices*nb_vertex_attr) ;
   edge_index.resize(2*nb_total_edges) ;
   edge_attr.reserve(nb_total_edges*nb_edge_attr) ;
   y.reserve(batch_size*y_size) ;
   int i = 0 ;
   int vertex_offset = 0 ;
   int edge_offset = 0 ;
   for(auto const& g : graph_batch)
   {
        for(int j=0;j<g.m_nb_vertices;++j)
           batch[vertex_offset+j] = i ;

        for(auto v : g.m_x)
        {
            x.push_back(v) ;
        }
        for(int j=0;j<g.m_nb_edges;++j)
        {
            edge_index[edge_offset+j] = vertex_offset+ g.m_edge_index[j] ;
            edge_index[nb_total_edges+edge_offset+j] = vertex_offset+ g.m_edge_index[g.m_nb_edges+j] ;
        }
        for(auto v : g.m_edge_attr)
        {
            edge_attr.push_back(v) ;
        }
        for(auto v : g.m_y)
        {
            y.push_back(v) ;
        }
        ++i ;
        vertex_offset += g.m_nb_vertices ;
        edge_offset   += g.m_nb_edges ;
   }
   {
       std::vector<int64_t> dims = { nb_total_vertices } ;
       torch::TensorOptions options(torch::kInt64);
       pt_graph.m_batch = torch::from_blob(batch.data(), torch::IntList(dims), options).clone() ;
   }
   {
       std::vector<int64_t> dims = { nb_total_vertices, nb_vertex_attr};
       torch::TensorOptions options(torch::kFloat64);
       pt_graph.m_x = torch::from_blob(x.data(), torch::IntList(dims), options).clone();
   }
   {
       std::vector<int64_t> dims = { 2, nb_total_edges};
       torch::TensorOptions options(torch::kInt64);
       pt_graph.m_edge_index = torch::from_blob(edge_index.data(), torch::IntList(dims), options).clone();
   }
   {
       std::vector<int64_t> dims = { nb_total_edges, nb_edge_attr};
       torch::TensorOptions options(torch::kFloat64);
       pt_graph.m_edge_attr = torch::from_blob(edge_attr.data(), torch::IntList(dims), options).clone();
   }
   {
       std::vector<int64_t> dims = { batch_size, y_size};
       torch::TensorOptions options(torch::kFloat64);
       pt_graph.m_y = torch::from_blob(y.data(), torch::IntList(dims), options).clone();
   }
   Graph graph = {nb_total_vertices, nb_total_edges, nb_vertex_attr, nb_edge_attr, y_size, batch, x, edge_index, edge_attr, y } ;
   std::cout<< graph;
}

void assignPTGraphToOnes(PTGraph& pt_graph, int64_t dim0, int64_t dim1)
{
   std::vector<int64_t> dims = {dim0, dim1};
   torch::TensorOptions options(torch::kFloat64);
   pt_graph.m_x = torch::ones(dims,options).clone() ;
}


void assignPTGraphToRandn(PTGraph& pt_graph, int64_t dim0, int64_t dim1)
{
   std::vector<int64_t> dims = {1, dim0, dim1};
   torch::TensorOptions options(torch::kFloat64);
   pt_graph.m_x = torch::randn(dims,options).clone() ;
}

