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
template<typename ValueT>
struct PredictionT ;
}

torch::jit::script::Module read_model(std::string, bool);

std::vector<ml4cfd::PredictionT<PTGraph::value_type>>
infer(torch::jit::script::Module model,
      std::vector<PTGraph>& graphs,
      bool usegpu,
      int nb_args,
      int batch_size) ;

std::vector<ml4cfd::PredictionT<PTGraphT<float,int64_t>::value_type>>
infer(torch::jit::script::Module model,
      std::vector<PTGraphT<float,int64_t>>& graphs,
      bool usegpu,
      int nb_args,
      int batch_size) ;
