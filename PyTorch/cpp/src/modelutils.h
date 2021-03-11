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

template<typename ValueT=double>
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
  oss<<"Prediction : dim("<<prediction.m_dim0<<","<<prediction.m_dim1<<")"<<std::endl ;
  oss<<"\t["<<std::endl ;
  int k = 0 ;
  for(int i=0;i<prediction.m_dim0;++i)
  {
      oss<<"[";
      for(int j=0;j<prediction.m_dim1;++j)
         oss<<prediction.m_values[k++]<<" ";
      oss<<"]"<<std::endl ;
  }
  oss<<"]"<<std::endl ;
}

torch::jit::script::Module read_model(std::string, bool);

std::vector<PredictionT<PTGraph::value_type>>
infer(torch::jit::script::Module model,
      std::vector<PTGraph>& graphs,
      bool usegpu,
      int opt) ;
