#include "graph.h"
#include "internal/graphutils.h"
#include "internal/cnpy.h"
//#include <filesystem>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace ml4cfd {

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

template<typename value_type, typename index_type>
void loadFromFileT(GraphT<value_type,index_type>& graph,std::string const& dir_path)
{

  boost::filesystem::path root_path(dir_path.c_str());

  std::cout<<"LOAD FILE : "<<(root_path / "sol.npy").string()<<std::endl;
  auto sol = cnpy::npy_load((root_path / "sol.npy").string()) ;
  std::cout<<"SOL DIMS : "<<sol.shape.size()<<" "<<sol.shape[0]<<std::endl ;

  std::cout<<"LOAD FILE : "<<(root_path / "edge_ij.npy").string()<<std::endl;
  auto edge_ij = cnpy::npy_load((root_path / "edge_ij.npy").string()) ;
  std::cout<<"INIDICES DIMS : "<<edge_ij.shape.size()<<" "<<edge_ij.shape[0]<<" "<<edge_ij.shape[1]<<std::endl ;

  std::cout<<"LOAD FILE : "<<(root_path / "edge_attr.npy").string()<<std::endl;
  auto edge_attr = cnpy::npy_load((root_path / "edge_attr.npy").string()) ;
  std::cout<<"VALUES DIMS : "<<edge_attr.shape.size()<<" "<<edge_attr.shape[0]<<" "<<edge_attr.shape[1]<<std::endl ;

  std::cout<<"LOAD FILE : "<<(root_path / "prb_data.npy").string()<<std::endl;
  auto prb_data = cnpy::npy_load((root_path / "prb_data.npy").string()) ;
  std::cout<<"VALUES DIMS : "<<prb_data.shape.size()<<" "<<prb_data.shape[0]<<" "<<prb_data.shape[1]<<std::endl ;

  std::cout<<"LOAD FILE : "<<(root_path / "pos.npy").string()<<std::endl;
  auto pos = cnpy::npy_load((root_path / "pos.npy").string()) ;
  std::cout<<"VALUES DIMS : "<<pos.shape.size()<<" "<<pos.shape[0]<<" "<<pos.shape[1]<<std::endl ;

  std::cout<<"LOAD FILE : "<<(root_path / "tags.npy").string()<<std::endl;
  auto tags = cnpy::npy_load((root_path / "tags.npy").string()) ;
  std::cout<<"VALUES DIMS : "<<tags.shape.size()<<" "<<tags.shape[0]<<" "<<tags.shape[1]<<std::endl ;

    {
        graph.m_nb_vertices = sol.shape[0] ;
        graph.m_nb_vertex_attr = 1 ;
        std::size_t size = graph.m_nb_vertices*graph.m_nb_vertex_attr ;
        graph.m_x.resize(size) ;
        graph.m_x.assign( sol.data<value_type>(),sol.data<value_type>()+size) ;
        std::cout<<"NB VERTICES : "<<graph.m_nb_vertices<<std::endl;
        std::cout<<"NB VERTEX ATTR : "<<graph.m_nb_vertex_attr<<std::endl;
        //for(int i=0;i<20;++i)
        //  std::cout<<"X["<<i<<"]"<<graph.m_x[i]<<std::endl ;
    }

    {
        std::cout<<"LOAD E : "<<edge_ij.shape[0]<<" "<<edge_ij.shape[1]<<std::endl ;
        graph.m_nb_edges = edge_ij.shape[1] ;
        graph.m_edge_index.resize(2*graph.m_nb_edges) ;
        graph.m_edge_index.assign(edge_ij.data<index_type>(),edge_ij.data<index_type>()+2*graph.m_nb_edges) ;
        //for(int i=0;i<graph.m_nb_edges;++i)
        //  std::cout<<"EDGE INDEX["<<i<<"]"<<graph.m_edge_index[i]<<" "<<graph.m_edge_index[graph.m_nb_edges+i]<<std::endl ;
    }

    {
      graph.m_nb_edge_attr = edge_attr.shape[1] ;
      std::size_t size = graph.m_nb_edges*graph.m_nb_edge_attr;
      std::cout<<"LOAD EDGE ATTR : "<<size<<std::endl ;
      graph.m_edge_attr.resize(size) ;
      graph.m_edge_attr.assign(edge_attr.data<value_type>(),edge_attr.data<value_type>()+size) ;
    }

    {
      std::size_t size = prb_data.shape[0] * prb_data.shape[1] ;
      std::cout<<"LOAD Y : "<<size<<std::endl ;
      graph.m_y_size = prb_data.shape[1] ;
      graph.m_y.resize(size) ;
      graph.m_y.assign( prb_data.data<value_type>(),prb_data.data<value_type>()+size) ;
   }

    {
      std::size_t size = pos.shape[0] * pos.shape[1] ;
      std::cout<<"LOAD POS : "<<size<<std::endl ;
      graph.m_pos.resize(size) ;
      graph.m_pos.assign( pos.data<value_type>(),pos.data<value_type>()+size) ;
      graph.m_dim = pos.shape[1] ;
    }

    {
      std::size_t size = tags.shape[0] ;
      std::cout<<"LOAD TAGS : "<<size<<std::endl ;
      graph.m_tags.resize(size) ;
      graph.m_tags.assign( tags.data<int>(),tags.data<int>()+size) ;
    }
}

template<typename value_type, typename index_type>
void loadFromJsonFileT(GraphT<value_type,index_type>& graph,std::string const& file_path, std::string b_name)
{
    namespace pt = boost::property_tree;

    // Create a root
    pt::ptree root;

    // Load the json file in this ptree
    pt::read_json(file_path, root);
    {
        graph.m_nb_vertices = 0 ;
        int count = 0 ;

        std::cout<<"LOAD X"<<std::endl ;
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
        std::cout<<"NB VERTICES : "<<graph.m_nb_vertices<<std::endl;
        std::cout<<"NB VERTEX ATTR : "<<graph.m_nb_vertex_attr<<std::endl;
    }

    {
        std::cout<<"LOAD E"<<std::endl ;
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
        std::cout<<"LOAD EDGE ATTR"<<std::endl ;
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
        std::cout<<"LOAD "<<b_name<<std::endl ;
        for (pt::ptree::value_type &y : root.get_child(b_name.c_str()))
        {
            for (pt::ptree::value_type &d : y.second)
            {
               graph.m_y.push_back(d.second.get_value<Graph::value_type>()) ;
            }
        }
        graph.m_y_size = graph.m_y.size()/graph.m_nb_vertices ;
    }
    {
        std::cout<<"LOAD POS"<<std::endl ;
        auto i_pos = root.find("pos");
        if(root.not_found() != i_pos)
        {
          for (pt::ptree::value_type &y : root.get_child("pos"))
          {
              for (pt::ptree::value_type &d : y.second)
              {
                 graph.m_pos.push_back(d.second.get_value<Graph::value_type>()) ;
              }
          }
          graph.m_dim = graph.m_pos.size()/graph.m_nb_vertices ;
        }
    }
    {
      std::cout<<"LOAD TAGS"<<std::endl ;
      auto i_tags = root.find("tags");
      if(root.not_found() != i_tags)
      {
        graph.m_tags.resize(0) ;
        graph.m_tags.reserve(graph.m_nb_vertices) ;
        for (pt::ptree::value_type &x : root.get_child("tags"))
        {
            for (pt::ptree::value_type &d : x.second)
            {
               graph.m_tags.push_back(d.second.get_value<int>()) ;
            }
        }
      }
    }
    {
      std::cout<<"LOAD AIJ"<<std::endl ;
      auto i_aij = root.find("aij");
      if(root.not_found() != i_aij)
      {
        graph.m_aij.resize(0) ;
        graph.m_aij.reserve(graph.m_nb_edges) ;
        for (pt::ptree::value_type &x : root.get_child("aij"))
        {
            for (pt::ptree::value_type &d : x.second)
            {
               graph.m_aij.push_back(d.second.get_value<Graph::value_type>()) ;
            }
        }
      }
    }
}


void loadFromFile(Graph& graph,std::string const& dir_path)
{
  loadFromFileT<double>(graph,dir_path) ;
}


void loadFromFile(GraphT<float,int64_t>& graph,std::string const& dir_path)
{
  loadFromFileT<float,int64_t>(graph,dir_path) ;
}

void loadFromJsonFile(Graph& graph,std::string const& dir_path, std::string b_name)
{
  loadFromJsonFileT<double>(graph,dir_path,b_name) ;
}


void loadFromJsonFile(GraphT<float,int64_t>& graph,std::string const& dir_path, std::string b_name)
{
  loadFromJsonFileT<float,int64_t>(graph,dir_path,b_name) ;
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
  return ostream;
}

void convertGraph2PTGraph(Graph& graph, PTGraph& pt_graph)
{
   {
       torch::TensorOptions options(torch::kInt64);
       pt_graph.m_batch = torch::zeros(graph.m_nb_vertices,options).clone() ;
   }
   {
       std::vector<int64_t> dims = {(int64_t)graph.m_nb_vertices, (int64_t)graph.m_nb_vertex_attr};
       torch::TensorOptions options(torch::kFloat64);
       pt_graph.m_x = torch::from_blob(graph.m_x.data(), torch::IntList(dims), options).clone();
   }
   {
       std::vector<int64_t> dims = {2, (int64_t)graph.m_nb_edges};
       torch::TensorOptions options(torch::kInt64);
       pt_graph.m_edge_index = torch::from_blob(graph.m_edge_index.data(), torch::IntList(dims), options).clone();
   }
   {
       std::vector<int64_t> dims = {(int64_t)graph.m_nb_edges,(int64_t)graph.m_nb_edge_attr};
       torch::TensorOptions options(torch::kFloat64);
       pt_graph.m_edge_attr = torch::from_blob(graph.m_edge_attr.data(), torch::IntList(dims), options).clone();
   }
   if(graph.m_y_size>0)
   {
       std::vector<int64_t> dims = {(int64_t)graph.m_nb_vertices, (int64_t)graph.m_y_size};
       torch::TensorOptions options(torch::kFloat64);
       pt_graph.m_y = torch::from_blob(graph.m_y.data(), torch::IntList(dims), options).clone();
   }
}


void convertGraph2PTGraph(GraphT<float,int64_t>& graph, PTGraphT<float,int64_t>& pt_graph)
{
   {
       torch::TensorOptions options(torch::kInt64);
       pt_graph.m_batch = torch::zeros(graph.m_nb_vertices,options).clone() ;
   }
   {
       std::vector<int64_t> dims = {(int64_t)graph.m_nb_vertices, (int64_t)graph.m_nb_vertex_attr};
       torch::TensorOptions options(torch::kFloat32);
       pt_graph.m_x = torch::from_blob(graph.m_x.data(), torch::IntList(dims), options).clone();
   }
   {
       std::vector<int64_t> dims = {2, (int64_t)graph.m_nb_edges};
       torch::TensorOptions options(torch::kInt64);
       pt_graph.m_edge_index = torch::from_blob(graph.m_edge_index.data(), torch::IntList(dims), options).clone();
   }
   {
       std::vector<int64_t> dims = {(int64_t)graph.m_nb_edges,(int64_t)graph.m_nb_edge_attr};
       torch::TensorOptions options(torch::kFloat32);
       pt_graph.m_edge_attr = torch::from_blob(graph.m_edge_attr.data(), torch::IntList(dims), options).clone();
   }
   if(graph.m_y_size>0)
   {
       std::vector<int64_t> dims = {(int64_t)graph.m_nb_vertices, 1};
       torch::TensorOptions options(torch::kFloat32);
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
   int64_t batch_size        = graph_batch.size() ;
   int64_t nb_total_vertices = 0 ;
   int64_t nb_total_edges    = 0 ;
   int64_t nb_vertex_attr    = graph_batch[0].m_nb_vertex_attr ;
   int64_t nb_edge_attr      = graph_batch[0].m_nb_edge_attr ;
   int64_t y_size            = graph_batch[0].m_y_size ;
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
   if(y_size>0)
   {
       std::vector<int64_t> dims = { nb_total_vertices, y_size};
       torch::TensorOptions options(torch::kFloat64);
       pt_graph.m_y = torch::from_blob(y.data(), torch::IntList(dims), options).clone();
   }
   //Graph graph = {nb_total_vertices, nb_total_edges, nb_vertex_attr, nb_edge_attr, y_size, batch, x, edge_index, edge_attr, y } ;
   //std::cout<< graph;
}

void convertGraph2PTGraph(Graph* begin, int batch_size, PTGraph& pt_graph)
{
   std::cout<<"CONVERT BATCH TO PTGRAPH : "<<batch_size<<std::endl ;
   std::vector<int64_t>           batch;
   std::vector<Graph::value_type> x ;
   std::vector<int64_t>           edge_index;
   std::vector<Graph::value_type> edge_attr ;
   std::vector<Graph::value_type> y ;
   int64_t nb_total_vertices = 0 ;
   int64_t nb_total_edges    = 0 ;
   int64_t nb_vertex_attr    = (*begin).m_nb_vertex_attr ;
   int64_t nb_edge_attr      = (*begin).m_nb_edge_attr ;
   int64_t y_size            = (*begin).m_y_size ;
   Graph* ptr = begin ;
   for(int i=0;i<batch_size;++i)
   {
       Graph const& g = *(ptr++) ;
       nb_total_vertices += g.m_nb_vertices ;
       nb_total_edges += g.m_nb_edges ;
       assert(g.m_nb_vertex_attr == nb_vertex_attr) ;
       assert(g.m_nb_edge_attr == nb_edge_attr) ;
   }
   batch.resize(nb_total_vertices) ;
   x.reserve(nb_total_vertices*nb_vertex_attr) ;
   edge_index.resize(2*nb_total_edges) ;
   edge_attr.reserve(nb_total_edges*nb_edge_attr) ;
   y.reserve(nb_total_vertices*y_size) ;
   int vertex_offset = 0 ;
   int edge_offset = 0 ;

   ptr = begin ;

   for(int i=0;i<batch_size;++i)
   {
        Graph const& g = *(ptr++) ;
        for(int j=0;j<g.m_nb_vertices;++j)
           batch[vertex_offset+j] = i ;
        {
          //int icount = 0 ;
          for(auto v : g.m_x)
          {
              x.push_back(v) ;
              //std::cout<<"x["<<icount<<"]"<<v<<std::endl ;
          }
        }
        {
          //int icount = 0 ;
          for(int j=0;j<g.m_nb_edges;++j)
          {
              edge_index[edge_offset+j] = vertex_offset+ g.m_edge_index[j] ;
              edge_index[nb_total_edges+edge_offset+j] = vertex_offset+ g.m_edge_index[g.m_nb_edges+j] ;
          }
        }
        {
          //int icount = 0 ;
          for(auto v : g.m_edge_attr)
          {
              edge_attr.push_back(v) ;
              //std::cout<<"ea["<<icount<<"]"<<v<<std::endl ;
          }
        }
        {
          //int icount = 0 ;
          for(auto v : g.m_y)
          {
              y.push_back(v) ;
              //std::cout<<"y["<<icount<<"]"<<v<<std::endl ;
          }
        }
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
       /*
       std::cout<<"X[" ;
       for( auto v : x )
         std::cout<<"["<<v<<"]"<<std::endl ;
       std::cout<<"]"<<std::endl ;
       */

   }
   {
       std::vector<int64_t> dims = { 2, nb_total_edges};
       torch::TensorOptions options(torch::kInt64);
       pt_graph.m_edge_index = torch::from_blob(edge_index.data(), torch::IntList(dims), options).clone();
       /*
       std::cout<<"EDGE[" ;
       for( auto v : edge_index )
         std::cout<<"["<<v<<"]"<<std::endl ;
       std::cout<<"]"<<std::endl ;
       */

   }
   {
       std::vector<int64_t> dims = { nb_total_edges, nb_edge_attr};
       torch::TensorOptions options(torch::kFloat64);
       pt_graph.m_edge_attr = torch::from_blob(edge_attr.data(), torch::IntList(dims), options).clone();
       /*
       std::cout<<"EDGE ATTR[" ;
       for( auto v : edge_attr )
         std::cout<<"["<<v<<"]"<<std::endl ;
       std::cout<<"]"<<std::endl ;
       */


   }
   if(y_size>0)
   {
       std::vector<int64_t> dims = { nb_total_vertices, y_size};
       torch::TensorOptions options(torch::kFloat64);
       pt_graph.m_y = torch::from_blob(y.data(), torch::IntList(dims), options).clone();
       /*
       std::cout<<"PBR[" ;
       for( auto v : y )
         std::cout<<"["<<v<<"]"<<std::endl ;
       std::cout<<"]"<<std::endl ;
       */

   }
}

void updateGraph2PTGraphData(Graph* begin, int batch_size, PTGraph& pt_graph)
{
   std::cout<<"UPDATE BATCH TO PTGRAPH DATA : "<<batch_size<<std::endl ;
   std::vector<Graph::value_type> x ;
   std::vector<Graph::value_type> y ;
   int64_t nb_total_vertices = 0 ;
   int64_t nb_total_edges    = 0 ;
   int64_t nb_vertex_attr    = (*begin).m_nb_vertex_attr ;
   int64_t y_size            = (*begin).m_y_size ;
   Graph* ptr = begin ;
   for(int i=0;i<batch_size;++i)
   {
       Graph const& g = *(ptr++) ;
       nb_total_vertices += g.m_nb_vertices ;
   }
   x.reserve(nb_total_vertices*nb_vertex_attr) ;
   y.reserve(nb_total_vertices*y_size) ;

   ptr = begin ;
   for(int i=0;i<batch_size;++i)
   {
        Graph const& g = *(ptr++) ;
        {
          for(auto v : g.m_x)
          {
              x.push_back(v) ;
          }
        }
        {
          for(auto v : g.m_y)
          {
              y.push_back(v) ;
          }
        }
   }
   {
       std::vector<int64_t> dims = { nb_total_vertices, nb_vertex_attr};
       torch::TensorOptions options(torch::kFloat64);
       pt_graph.m_x = torch::from_blob(x.data(), torch::IntList(dims), options).clone();
   }
   {
       std::vector<int64_t> dims = { nb_total_vertices, y_size};
       torch::TensorOptions options(torch::kFloat64);
       pt_graph.m_y = torch::from_blob(y.data(), torch::IntList(dims), options).clone();
   }
}



void convertGraph2PTGraph(std::vector<GraphT<float,int64_t>>& graph_batch, PTGraphT<float,int64_t>& pt_graph)
{
   std::cout<<"CONVERT BATCH TO PTGRAPH : "<<graph_batch.size()<<std::endl ;
   std::vector<int64_t>           batch;
   std::vector<GraphT<float,int64_t>::value_type> x ;
   std::vector<int64_t>           edge_index;
   std::vector<GraphT<float,int64_t>::value_type> edge_attr ;
   std::vector<GraphT<float,int64_t>::value_type> y ;
   int64_t batch_size        = graph_batch.size() ;
   int64_t nb_total_vertices = 0 ;
   int64_t nb_total_edges    = 0 ;
   int64_t nb_vertex_attr    = graph_batch[0].m_nb_vertex_attr ;
   int64_t nb_edge_attr      = graph_batch[0].m_nb_edge_attr ;
   int64_t y_size            = graph_batch[0].m_y_size ;
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
   int index = 0 ;
   int vertex_offset = 0 ;
   int edge_offset = 0 ;
   for(auto const& g : graph_batch)
   {
        for(int j=0;j<g.m_nb_vertices;++j)
           batch[vertex_offset+j] = index++ ;

        {
          for(auto v : g.m_x)
          {
              x.push_back(v) ;
          }
        }
        for(int j=0;j<g.m_nb_edges;++j)
        {
            edge_index[edge_offset+j] = vertex_offset+ g.m_edge_index[j] ;
            edge_index[nb_total_edges+edge_offset+j] = vertex_offset+ g.m_edge_index[g.m_nb_edges+j] ;
        }
        {
          for(auto v : g.m_edge_attr)
          {
              edge_attr.push_back(v) ;
          }
        }
        {
          for(auto v : g.m_y)
          {
              y.push_back(v) ;
          }
        }
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
       torch::TensorOptions options(torch::kFloat32);
       pt_graph.m_x = torch::from_blob(x.data(), torch::IntList(dims), options).clone();
   }
   {
       std::vector<int64_t> dims = { 2, nb_total_edges};
       torch::TensorOptions options(torch::kInt64);
       pt_graph.m_edge_index = torch::from_blob(edge_index.data(), torch::IntList(dims), options).clone();
   }
   {
       std::vector<int64_t> dims = { nb_total_edges, nb_edge_attr};
       torch::TensorOptions options(torch::kFloat32);
       pt_graph.m_edge_attr = torch::from_blob(edge_attr.data(), torch::IntList(dims), options).clone();
   }
   if(y_size>0)
   {
       std::vector<int64_t> dims = { nb_total_vertices, y_size};
       torch::TensorOptions options(torch::kFloat32);
       pt_graph.m_y = torch::from_blob(y.data(), torch::IntList(dims), options).clone();
   }
}

void convertGraph2PTGraph(GraphT<float,int64_t>* begin, int batch_size, PTGraphT<float,int64_t>& pt_graph)
{
   //std::cout<<"CONVERT BATCH TO PTGRAPH : "<<batch_size<<std::endl ;
   std::vector<int64_t>           batch;
   std::vector<GraphT<float,int64_t>::value_type> x ;
   std::vector<int64_t>           edge_index;
   std::vector<GraphT<float,int64_t>::value_type> edge_attr ;
   std::vector<GraphT<float,int64_t>::value_type> y ;
   int64_t nb_total_vertices = 0 ;
   int64_t nb_total_edges    = 0 ;
   int64_t nb_vertex_attr    = (*begin).m_nb_vertex_attr ;
   int64_t nb_edge_attr      = (*begin).m_nb_edge_attr ;
   int64_t y_size            = (*begin).m_y_size ;
   GraphT<float,int64_t>* ptr = begin ;
   for(int i=0;i<batch_size;++i)
   {
       auto const& g = *(ptr++) ;
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
   int index = 0 ;
   int vertex_offset = 0 ;
   int edge_offset = 0 ;

   ptr = begin ;
   for(int i=0;i<batch_size;++i)
   {
        //std::cout<<"BATCH["<<i<<"]"<<std::endl ;
        auto const& g = *(ptr++) ;
        for(int j=0;j<g.m_nb_vertices;++j)
           batch[vertex_offset+j] = index++ ;
        //std::cout<<"GRAPH X SIZE :"<<g.m_x.size()<<std::endl ;
        for(auto v : g.m_x)
        {
            x.push_back(v) ;
        }

        for(int j=0;j<g.m_nb_edges;++j)
        {
            edge_index[edge_offset+j] = vertex_offset+ g.m_edge_index[j] ;
            edge_index[nb_total_edges+edge_offset+j] = vertex_offset+ g.m_edge_index[g.m_nb_edges+j] ;
            //std::cout<<"EDGE INDEX["<<j<<"]:("<<g.m_edge_index[j]<<" "<<g.m_edge_index[g.m_nb_edges+j]<<")("<<edge_index[edge_offset+j]<<" "<<edge_index[nb_total_edges+edge_offset+j]<<")"<<std::endl ;
        }

        for(auto v : g.m_edge_attr)
        {
            edge_attr.push_back(v) ;
        }
        //std::cout<<"GRAPH Y SIZE :"<<g.m_y.size()<<std::endl ;
        for(auto v : g.m_y)
        {
            y.push_back(v) ;
        }
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
       torch::TensorOptions options(torch::kFloat32);
       pt_graph.m_x = torch::from_blob(x.data(), torch::IntList(dims), options).clone();
       pt_graph.m_x_is_updated          = true ;


       /*
       std::cout<<"X[" ;
       for( auto v : x )
         std::cout<<"["<<v<<"]"<<std::endl ;
       std::cout<<"]"<<std::endl ;
       */
   }
   {
       std::vector<int64_t> dims = { 2, nb_total_edges};
       torch::TensorOptions options(torch::kInt64);
       pt_graph.m_edge_index = torch::from_blob(edge_index.data(), torch::IntList(dims), options).clone();
       pt_graph.m_edge_index_is_updated = true;

       /*
       std::cout<<"EDGE[" ;
       for( int i=0;i<nb_total_edges;++i )
         std::cout<<"E["<<i<<"] : ["<<edge_index[i]<<","<<edge_index[i+nb_total_edges]<<"]"<<std::endl ;
       std::cout<<"]"<<std::endl ;
       */
   }
   {
       std::vector<int64_t> dims = { nb_total_edges, nb_edge_attr};
       torch::TensorOptions options(torch::kFloat32);
       pt_graph.m_edge_attr = torch::from_blob(edge_attr.data(), torch::IntList(dims), options).clone();
       pt_graph.m_edge_attr_is_updated  = true ;

       /*
       std::cout<<"EDGE ATTR[" ;
       for( int i=0;i<nb_total_edges;++i )
       {
         std::cout<<"[";
         for(int j=0;j<nb_edge_attr;++j)
           std::cout<<edge_attr[nb_edge_attr*i+j];
         std::cout<<"]"<<std::endl ;
       }
       std::cout<<"]"<<std::endl ;
       */
   }
   if(y_size>0)
   {
       std::vector<int64_t> dims = { nb_total_vertices, y_size};
       torch::TensorOptions options(torch::kFloat32);
       pt_graph.m_y = torch::from_blob(y.data(), torch::IntList(dims), options).clone();
       pt_graph.m_y_is_updated          = true ;

       /*
       std::cout<<"PBR[" ;
       for( auto v : y )
         std::cout<<"["<<v<<"]"<<std::endl ;
       std::cout<<"]"<<std::endl ;
       */
   }
}

void updateGraph2PTGraphData(GraphT<float,int64_t>* begin, int batch_size, PTGraphT<float,int64_t>& pt_graph)
{
   std::cout<<"UPDATE BATCH TO PTGRAPH : "<<batch_size<<" "<<pt_graph.m_y_is_updated<<std::endl ;
   std::vector<GraphT<float,int64_t>::value_type> x ;
   std::vector<GraphT<float,int64_t>::value_type> y ;
   int64_t nb_total_vertices = 0 ;
   int64_t nb_vertex_attr    = (*begin).m_nb_vertex_attr ;
   int64_t nb_edge_attr      = (*begin).m_nb_edge_attr ;
   int64_t y_size            = (*begin).m_y_size ;
   GraphT<float,int64_t>* ptr = begin ;
   for(int i=0;i<batch_size;++i)
   {
       auto const& g = *(ptr++) ;
       nb_total_vertices += g.m_nb_vertices ;
   }
   x.reserve(nb_total_vertices*nb_vertex_attr) ;
   y.reserve(nb_total_vertices*y_size) ;
   int i = 0 ;
   ptr = begin ;
   for(int i=0;i<batch_size;++i)
   {
        auto const& g = *(ptr++) ;
        if(!pt_graph.m_x_is_updated)
        {
          for(auto v : g.m_x)
          {
              x.push_back(v) ;
          }
        }
        if(!pt_graph.m_y_is_updated)
        {
          for(auto v : g.m_y)
          {
              y.push_back(v) ;
          }
        }
   }
   if(!pt_graph.m_x_is_updated)
   {
       std::cout<<"UPDATE PTGRAPH X"<<std::endl ;
       std::vector<int64_t> dims = { nb_total_vertices, nb_vertex_attr};
       torch::TensorOptions options(torch::kFloat32);
       pt_graph.m_x = torch::from_blob(x.data(), torch::IntList(dims), options).clone();
       pt_graph.m_x_device = torch::Device(torch::kCPU) ;
       pt_graph.m_x_is_updated = true ;
   }
   if(!pt_graph.m_y_is_updated)
   {
       std::cout<<"UPDATE PTGRAPH Y"<<std::endl ;
       std::vector<int64_t> dims = { nb_total_vertices, y_size};
       torch::TensorOptions options(torch::kFloat32);
       pt_graph.m_y = torch::from_blob(y.data(), torch::IntList(dims), options).clone();
       pt_graph.m_y_device = torch::Device(torch::kCPU) ;
       pt_graph.m_y_is_updated = true ;
   }
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

template<typename ValueT, typename IndexType, typename ONNXIndexType>
void convertGraph2PTGraphT(GraphT<ValueT,IndexType>* begin, int batch_size, ONNXGraphT<ValueT,ONNXIndexType>& onnx_graph)
{
   //std::cout<<"CONVERT BATCH TO PTGRAPH : "<<batch_size<<std::endl ;
   int64_t nb_total_vertices = 0 ;
   int64_t nb_total_edges    = 0 ;
   int64_t nb_vertex_attr    = (*begin).m_nb_vertex_attr ;
   int64_t nb_edge_attr      = (*begin).m_nb_edge_attr ;
   int64_t y_size            = (*begin).m_y_size ;
   GraphT<ValueT,IndexType>* ptr = begin ;
   for(int i=0;i<batch_size;++i)
   {
       auto const& g = *(ptr++) ;
       nb_total_vertices += g.m_nb_vertices ;
       nb_total_edges += g.m_nb_edges ;
       assert(g.m_nb_vertex_attr == nb_vertex_attr) ;
       assert(g.m_nb_edge_attr == nb_edge_attr) ;
   }
   onnx_graph.m_total_nb_vertices = nb_total_vertices ;
   onnx_graph.m_total_nb_edges = nb_total_edges ;
   onnx_graph.m_batch.resize(nb_total_vertices) ;
   onnx_graph.m_x.reserve(nb_total_vertices*nb_vertex_attr) ;
   onnx_graph.m_edge_index.resize(2*nb_total_edges) ;
   onnx_graph.m_edge_attr.reserve(nb_total_edges*nb_edge_attr) ;
   onnx_graph.m_y.reserve(batch_size*y_size) ;
   int index = 0 ;
   int vertex_offset = 0 ;
   int edge_offset = 0 ;

   ptr = begin ;
   for(int i=0;i<batch_size;++i)
   {
        //std::cout<<"BATCH["<<i<<"]"<<std::endl ;
        auto const& g = *(ptr++) ;
        for(int j=0;j<g.m_nb_vertices;++j)
          onnx_graph.m_batch[vertex_offset+j] = index++ ;
        //std::cout<<"GRAPH X SIZE :"<<g.m_x.size()<<std::endl ;
        for(auto v : g.m_x)
        {
          onnx_graph.m_x.push_back(v) ;
        }

        for(int j=0;j<g.m_nb_edges;++j)
        {
          onnx_graph.m_edge_index[edge_offset+j] = vertex_offset+ g.m_edge_index[j] ;
          onnx_graph.m_edge_index[nb_total_edges+edge_offset+j] = vertex_offset+ g.m_edge_index[g.m_nb_edges+j] ;
            //std::cout<<"EDGE INDEX["<<j<<"]:("<<g.m_edge_index[j]<<" "<<g.m_edge_index[g.m_nb_edges+j]<<")("<<edge_index[edge_offset+j]<<" "<<edge_index[nb_total_edges+edge_offset+j]<<")"<<std::endl ;
        }

        for(auto v : g.m_edge_attr)
        {
          onnx_graph.m_edge_attr.push_back(v) ;
        }
        //std::cout<<"GRAPH Y SIZE :"<<g.m_y.size()<<std::endl ;
        for(auto v : g.m_y)
        {
          onnx_graph.m_y.push_back(v) ;
        }
        vertex_offset += g.m_nb_vertices ;
        edge_offset   += g.m_nb_edges ;
   }
}

template<typename ValueT, typename IndexType, typename ONNXIndexType>
void updateGraph2PTGraphDataT(GraphT<ValueT,IndexType>* begin, int batch_size, ONNXGraphT<ValueT,ONNXIndexType>& onnx_graph)
{
   std::cout<<"UPDATE BATCH TO ONNXGRAPH : "<<batch_size<<" "<<onnx_graph.m_y_is_updated<<std::endl ;
   int64_t nb_vertex_attr    = (*begin).m_nb_vertex_attr ;
   int64_t nb_edge_attr      = (*begin).m_nb_edge_attr ;
   int64_t y_size            = (*begin).m_y_size ;
   if(!onnx_graph.m_x_is_updated)
   {
     onnx_graph.m_x.resize(0) ;
   }
   if(!onnx_graph.m_y_is_updated)
   {
     onnx_graph.m_y.resize(0) ;
   }
   GraphT<ValueT,IndexType>* ptr = begin ;
   for(int i=0;i<batch_size;++i)
   {
        auto const& g = *(ptr++) ;
        if(!onnx_graph.m_x_is_updated)
        {
          for(auto v : g.m_x)
          {
            onnx_graph.m_x.push_back(v) ;
          }
        }
        if(!onnx_graph.m_y_is_updated)
        {
          for(auto v : g.m_y)
          {
            onnx_graph.m_y.push_back(v) ;
          }
        }
   }
}

void convertGraph2PTGraph(GraphT<float,int64_t>* begin, int batch_size, ONNXGraphT<float,int64_t>& onnx_graph)
{
  convertGraph2PTGraphT<float,int64_t,int64_t>(begin,batch_size,onnx_graph) ;
}

void convertGraph2PTGraph(GraphT<float,int64_t>* begin, int batch_size, ONNXGraphT<float,int>& onnx_graph)
{
  convertGraph2PTGraphT<float,int64_t,int>(begin,batch_size,onnx_graph) ;
}

void convertGraph2PTGraph(GraphT<double,int64_t>* begin, int batch_size, ONNXGraphT<double,int64_t>& onnx_graph)
{
  convertGraph2PTGraphT<double,int64_t,int64_t>(begin,batch_size,onnx_graph) ;
}

void convertGraph2PTGraph(GraphT<double,int64_t>* begin, int batch_size, ONNXGraphT<double,int>& onnx_graph)
{
  convertGraph2PTGraphT<double,int64_t,int>(begin,batch_size,onnx_graph) ;
}


void updateGraph2PTGraphData(GraphT<float,int64_t>* begin, int batch_size, ONNXGraphT<float,int64_t>& onnx_graph)
{
  return updateGraph2PTGraphDataT<float,int64_t,int64_t>(begin,batch_size,onnx_graph) ;
}

void updateGraph2PTGraphData(GraphT<float,int64_t>* begin, int batch_size, ONNXGraphT<float,int>& onnx_graph)
{
  return updateGraph2PTGraphDataT<float,int64_t,int>(begin,batch_size,onnx_graph) ;
}

void updateGraph2PTGraphData(GraphT<double,int64_t>* begin, int batch_size, ONNXGraphT<double,int64_t>& onnx_graph)
{
  return updateGraph2PTGraphDataT<double,int64_t,int64_t>(begin,batch_size,onnx_graph) ;
}

void updateGraph2PTGraphData(GraphT<double,int64_t>* begin, int batch_size, ONNXGraphT<double,int>& onnx_graph)
{
  return updateGraph2PTGraphDataT<double,int64_t,int>(begin,batch_size,onnx_graph) ;
}


template<typename ValueT>
void convertGraph2PTGraphT(GraphT<ValueT,int64_t>* begin, int batch_size, TensorRTGraphT<ValueT,int64_t>& onnx_graph)
{
   //std::cout<<"CONVERT BATCH TO PTGRAPH : "<<batch_size<<std::endl ;
   std::vector<int64_t>           batch;
   int64_t nb_total_vertices = 0 ;
   int64_t nb_total_edges    = 0 ;
   int64_t nb_vertex_attr    = (*begin).m_nb_vertex_attr ;
   int64_t nb_edge_attr      = (*begin).m_nb_edge_attr ;
   int64_t y_size            = (*begin).m_y_size ;
   GraphT<ValueT,int64_t>* ptr = begin ;
   for(int i=0;i<batch_size;++i)
   {
       auto const& g = *(ptr++) ;
       nb_total_vertices += g.m_nb_vertices ;
       nb_total_edges += g.m_nb_edges ;
       assert(g.m_nb_vertex_attr == nb_vertex_attr) ;
       assert(g.m_nb_edge_attr == nb_edge_attr) ;
   }
   onnx_graph.m_total_nb_vertices = nb_total_vertices ;
   onnx_graph.m_total_nb_edges = nb_total_edges ;
   onnx_graph.m_batch.resize(nb_total_vertices) ;
   onnx_graph.m_x.reserve(nb_total_vertices*nb_vertex_attr) ;
   onnx_graph.m_edge_index.resize(2*nb_total_edges) ;
   onnx_graph.m_edge_attr.reserve(nb_total_edges*nb_edge_attr) ;
   onnx_graph.m_y.reserve(batch_size*y_size) ;
   int index = 0 ;
   int vertex_offset = 0 ;
   int edge_offset = 0 ;

   ptr = begin ;
   for(int i=0;i<batch_size;++i)
   {
        //std::cout<<"BATCH["<<i<<"]"<<std::endl ;
        auto const& g = *(ptr++) ;
        for(int j=0;j<g.m_nb_vertices;++j)
          onnx_graph.m_batch[vertex_offset+j] = index++ ;
        //std::cout<<"GRAPH X SIZE :"<<g.m_x.size()<<std::endl ;
        for(auto v : g.m_x)
        {
          onnx_graph.m_x.push_back(v) ;
        }

        for(int j=0;j<g.m_nb_edges;++j)
        {
          onnx_graph.m_edge_index[edge_offset+j] = vertex_offset+ g.m_edge_index[j] ;
          onnx_graph.m_edge_index[nb_total_edges+edge_offset+j] = vertex_offset+ g.m_edge_index[g.m_nb_edges+j] ;
            //std::cout<<"EDGE INDEX["<<j<<"]:("<<g.m_edge_index[j]<<" "<<g.m_edge_index[g.m_nb_edges+j]<<")("<<edge_index[edge_offset+j]<<" "<<edge_index[nb_total_edges+edge_offset+j]<<")"<<std::endl ;
        }

        for(auto v : g.m_edge_attr)
        {
          onnx_graph.m_edge_attr.push_back(v) ;
        }
        //std::cout<<"GRAPH Y SIZE :"<<g.m_y.size()<<std::endl ;
        for(auto v : g.m_y)
        {
          onnx_graph.m_y.push_back(v) ;
        }
        vertex_offset += g.m_nb_vertices ;
        edge_offset   += g.m_nb_edges ;
   }
}

template<typename ValueT>
void updateGraph2PTGraphDataT(GraphT<ValueT,int64_t>* begin, int batch_size, TensorRTGraphT<ValueT,int64_t>& onnx_graph)
{
   std::cout<<"UPDATE BATCH TO ONNXGRAPH : "<<batch_size<<" "<<onnx_graph.m_y_is_updated<<std::endl ;
   int64_t nb_vertex_attr    = (*begin).m_nb_vertex_attr ;
   int64_t nb_edge_attr      = (*begin).m_nb_edge_attr ;
   int64_t y_size            = (*begin).m_y_size ;
   if(!onnx_graph.m_x_is_updated)
   {
     onnx_graph.m_x.resize(0) ;
   }
   if(!onnx_graph.m_y_is_updated)
   {
     onnx_graph.m_y.resize(0) ;
   }
   GraphT<ValueT,int64_t>* ptr = begin ;
   for(int i=0;i<batch_size;++i)
   {
        auto const& g = *(ptr++) ;
        if(!onnx_graph.m_x_is_updated)
        {
          for(auto v : g.m_x)
          {
            onnx_graph.m_x.push_back(v) ;
          }
        }
        if(!onnx_graph.m_y_is_updated)
        {
          for(auto v : g.m_y)
          {
            onnx_graph.m_y.push_back(v) ;
          }
        }
   }
}

void convertGraph2PTGraph(GraphT<float,int64_t>* begin, int batch_size, TensorRTGraphT<float,int64_t>& onnx_graph)
{
  convertGraph2PTGraphT(begin,batch_size,onnx_graph) ;
}

void convertGraph2PTGraph(GraphT<double,int64_t>* begin, int batch_size, TensorRTGraphT<double,int64_t>& onnx_graph)
{
  convertGraph2PTGraphT(begin,batch_size,onnx_graph) ;
}

void updateGraph2PTGraphData(GraphT<float,int64_t>* begin, int batch_size, TensorRTGraphT<float,int64_t>& onnx_graph)
{
  return updateGraph2PTGraphDataT(begin,batch_size,onnx_graph) ;
}

void updateGraph2PTGraphData(GraphT<double,int64_t>* begin, int batch_size, TensorRTGraphT<double,int64_t>& onnx_graph)
{
  return updateGraph2PTGraphDataT(begin,batch_size,onnx_graph) ;
}

}
