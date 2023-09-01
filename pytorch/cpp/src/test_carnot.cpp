
#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <fstream>
#include <random>
#include <string>
#include <sstream>
#include <memory>


#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>

//#include <filesystem>
#include <boost/filesystem.hpp>

#include <torch/script.h>
#include <torch/serialize/tensor.h>
#include <torch/serialize.h>

#include "cnpy.h"

#include "utils/PerfCounterMng.h"
#include "carnot/PTFlash.h"

int main(int argc, char **argv)
{
    namespace po = boost::program_options;
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "produce help message")
    ("use-gpu",         po::value<int>()->default_value(0), "use gpu option")
    ("test-load-model", po::value<int>()->default_value(0), "test load model")
    ("test-inference",  po::value<int>()->default_value(0), "test inference")
    ("test-carnot",     po::value<int>()->default_value(0), "test carnot")
    ("model",           po::value<std::string>()->default_value("torch"), "inference model")
    ("test-data-id",    po::value<int>()->default_value(0), "test data id")
    ("batch-size",      po::value<int>()->default_value(1), "batch size")
    ("nb-comp",         po::value<int>()->default_value(2), "nb compo")
    ("output",          po::value<int>()->default_value(0), "output level")
    ("data-file",       po::value<std::string>(),           "data file path")
    ("result-file",     po::value<std::string>(),           "result file path")
    ("model-file",      po::value<std::string>(),           "model file path") ;


    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    bool use_gpu = vm["use-gpu"].as<int>() == 1 ;

    PerfCounterMng<std::string> perf_mng ;
    perf_mng.init("Torch:Prepare") ;
    perf_mng.init("Torch:Compute") ;
    perf_mng.init("Torch:Compute0") ;
    perf_mng.init("Torch:Compute1") ;
    perf_mng.init("Torch:Compute2") ;
    perf_mng.init("CAWF:Prepare") ;
    perf_mng.init("CAWF:Compute") ;
    perf_mng.init("CARNOT:Prepare") ;
    perf_mng.init("CARNOT:Compute") ;

    if(vm["test-load-model"].as<int>() == 1)
    {
      std::string model_path = vm["model-file"].as<std::string>();
      std::cout<<"TEST LOAD MODEL : "<<model_path<<std::endl ;
      torch::jit::script::Module model = torch::jit::load(model_path);
    }

    if(vm["test-inference"].as<int>() > 0 )
    {
       bool use_torch  = vm["model"].as<std::string>().compare("torch") == 0 ;
       bool use_cawf   = vm["model"].as<std::string>().compare("cawf") == 0 ;
       bool use_carnot = vm["model"].as<std::string>().compare("carnot") == 0 ;

       std::string data_path = vm["data-file"].as<std::string>();
       auto data = cnpy::npy_load(data_path) ;
       auto data_dims = data.shape.size() ;
       assert(data_dims == 2) ;

       auto nrows = data.shape[0] ;
       auto ncols = data.shape[1] ;


       int nb_comp = vm["nb-comp"].as<int>() ;
       std::cout<<"nb rows  ="<<nrows<<std::endl ;
       std::cout<<"nb cols  ="<<ncols<<std::endl ;
       std::cout<<"nb comps ="<<nb_comp<<std::endl ;

       assert(ncols >= 2 + nb_comp) ;

       int test_data_id = vm["test-data-id"].as<int>() ;
       int batch_size   = vm["batch-size"].as<int>() ;

       typedef double value_type ;
       std::size_t tensor_size = 2 + nb_comp ;
       std::vector<double> x(batch_size*tensor_size) ;
       std::cout<<"TEST DATA ID : "<<test_data_id<<std::endl ;
       int offset = 0 ;
       for(int i=0;i<batch_size;++i)
       {
           int id = (test_data_id+i)%nrows ;
           //std::cout<<"[P, T, Z ] : [";
           for(int ic=0;ic<tensor_size;++ic)
           {
              x[offset+ic] = data.data<value_type>()[ic*nrows+id] ;
              //std::cout<<x[offset + ic]<<",";
           }
           //std::cout<<"]"<<std::endl ;
           offset += tensor_size ;
       }

       std::string model_path = vm["model-file"].as<std::string>();
       std::cout<<"TEST INFERENCE : "<<model_path<<std::endl ;
       std::vector<bool>     unstable ;
       std::vector<double>   theta_v ;
       std::vector<double>   xi ;
       std::vector<double>   yi ;
       std::vector<double>   ki ;

       if(use_torch)
       {
         perf_mng.start("Torch:Prepare") ;
         torch::jit::script::Module model = torch::jit::load(model_path);

         torch::DeviceType device_type = use_gpu ? torch::kCUDA : torch::kCPU;
         torch::Device device(device_type);
         model.to(device);

         std::vector<torch::jit::IValue> inputs;
         std::vector<int64_t> dims = { batch_size, tensor_size};
         torch::TensorOptions options(torch::kFloat64);
         torch::Tensor input = torch::from_blob(x.data(), torch::IntList(dims), options).clone().to(device);
         inputs.push_back(input) ;
         perf_mng.stop("Torch:Prepare") ;

         std::cout<<"FORWARD : ";
         {
           perf_mng.start("Torch:Compute0") ;
           auto outputs = model.forward(inputs).toTuple();
           perf_mng.stop("Torch:Compute0") ;
         }
         {
           perf_mng.start("Torch:Compute1") ;
           auto outputs = model.forward(inputs).toTuple();
           perf_mng.stop("Torch:Compute1") ;
         }
         {
           perf_mng.start("Torch:Compute2") ;
           auto outputs = model.forward(inputs).toTuple();
           perf_mng.stop("Torch:Compute2") ;
         }
         perf_mng.start("Torch:Compute") ;
         auto outputs = model.forward(inputs).toTuple();
         perf_mng.stop("Torch:Compute") ;
         std::cout<<"AFTER FORWARD"<<std::endl ;

         torch::Device cpu_device(torch::kCPU);
         //auto cpu_outputs = outputs.to(cpu_device);
         {
             auto out = outputs->elements()[0].toTensor().to(cpu_device);
             torch::ArrayRef<int64_t> sizes = out.sizes();
             std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes.size()<<std::endl ;
             unstable.resize(sizes[0]) ;
             unstable.assign(out.data_ptr<bool>(),out.data_ptr<bool>() + sizes[0]) ;
         }
         {
             auto out = outputs->elements()[1].toTensor().to(cpu_device);
             //auto out_acc = out.accessor<double,1>() ;
             //std::cout<<"SIZE : "<<out_acc.size(0)<<std::endl ;
             torch::ArrayRef<int64_t> sizes = out.sizes();
             std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes[1]<<" "<<sizes.size()<<std::endl ;
             if(sizes[0]>0)
             {
                 theta_v.resize(sizes[0]) ;
                 theta_v.assign(out.data_ptr<double>(),out.data_ptr<double>() + sizes[0]);
             }
         }
         {
             auto out = outputs->elements()[2].toTensor().to(cpu_device);
             //auto out_acc = out.accessor<double,2>() ;
             //std::cout<<"SIZE : "<<out_acc.size(0)<<std::endl ;
             torch::ArrayRef<int64_t> sizes = out.sizes();
             std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes[1]<<" "<<sizes.size()<<std::endl ;
             if(sizes[0]>0)
              {
                  xi.resize(sizes[0]*sizes[1]) ;
                  xi.assign(out.data_ptr<double>(),out.data_ptr<double>() + sizes[0]*sizes[1]);
              }
         }
         {
             auto out = outputs->elements()[3].toTensor().to(cpu_device);
             torch::ArrayRef<int64_t> sizes = out.sizes();
             std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes[1]<<" "<<sizes.size()<<std::endl ;
             if(sizes[0]>0)
             {
                 yi.resize(sizes[0]*sizes[1]) ;
                 yi.assign(out.data_ptr<double>(),out.data_ptr<double>() + sizes[0]*sizes[1]);
             }
          }
         {
             auto out = outputs->elements()[4].toTensor().to(cpu_device);
             torch::ArrayRef<int64_t> sizes = out.sizes();
             std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes[1]<<" "<<sizes.size()<<std::endl ;
             if(sizes[0]>0)
             {
                 ki.resize(sizes[0]*sizes[1]) ;
                 ki.assign(out.data_ptr<double>(),out.data_ptr<double>() + sizes[0]*sizes[1]);
             }
         }
       }

       PTFlash ptflash(vm["output"].as<int>()) ;
       std::string prepare_phase ;
       std::string compute_phase ;
       if(use_cawf)
       {
           prepare_phase = "CAWF:Prepare" ;
           prepare_phase = "CAWF:Compute" ;
           ptflash.initCAWF(model_path,2,nb_comp) ;
           ptflash.startCompute(batch_size) ;
       }
       if(use_carnot)
       {
           prepare_phase = "CARNOT:Prepare" ;
           compute_phase = "CARNOT:Compute" ;
           std::vector<int> comp_uids = {74828, 74840, 74986, 106978, 109660, 110543, 100007, 124389, 7727379} ;
           ptflash.initCarnot(model_path,2,nb_comp,comp_uids) ;
           ptflash.startCompute(batch_size) ;
       }
       if(use_cawf || use_carnot)
       {
           perf_mng.start(prepare_phase) ;
           for(int i=0;i<batch_size;++i)
           {
                int id = (test_data_id+i)%nrows ;
                //std::cout<<"[P, T, Z ] : [";
                double P = data.data<value_type>()[id] ;
                double T = data.data<value_type>()[nrows+id] ;
                //std::cout<<P<<","<<T<<",";
                std::vector<double> zk(nb_comp) ;
                for(int ic=0;ic<nb_comp;++ic)
                {
                   zk[ic] = data.data<value_type>()[(2+ic)*nrows+id] ;
                   //std::cout<<zk[ic]<<",";
                }
                //std::cout<<"]"<<std::endl ;
                ptflash.asynchCompute(P,T,zk) ;
           }
           perf_mng.stop(prepare_phase) ;

           perf_mng.start(compute_phase) ;
           ptflash.endCompute() ;
           perf_mng.stop(compute_phase) ;
           if(vm["output"].as<int>() >0)
           {
             for(int i=0;i<batch_size;++i)
              {
                  std::vector<double> theta_v(2) ;
                  std::vector<double> xkp(2*nb_comp) ;
                  bool unstable = ptflash.getNextResult(theta_v,xkp) ;
                  std::cout<<"RES["<<i<<"] UNSTABLE:"<<unstable<<std::endl ;
                  if(unstable)
                  {
                    std::cout<<"             THETA_V : "<<theta_v[0]<<std::endl ;
                    std::cout<<"             LIQ XI : [";
                    for(int j=0;j<nb_comp;++j)
                      std::cout<<xkp[j]<<(j==nb_comp-1?" ]":" ");
                    std::cout<<std::endl ;
                    std::cout<<"             VAP YI : [";
                    for(int j=0;j<nb_comp;++j)
                      std::cout<<xkp[nb_comp+j]<<(j==nb_comp-1?" ]":" ");
                    std::cout<<std::endl ;
                  }
              }
          }
       }
       else if(vm["output"].as<int>() >0)
       {
         offset = 0 ;
         int current_id = 0 ;
         for(int i=0;i<batch_size;++i)
         {
             bool value = unstable[i] ;
             std::cout<<"RES["<<i<<"] UNSTABLE:"<<value<<std::endl ;
             if(value)
             {
               std::cout<<"             THETA_V : "<<theta_v[current_id++]<<std::endl ;
               std::cout<<"             LIQ XI : [";
               for(int j=0;j<nb_comp;++j)
                 std::cout<<xi[offset+j]<<(j==nb_comp-1?" ]":" ");
               std::cout<<std::endl ;
               std::cout<<"             VAP YI : [";
               for(int j=0;j<nb_comp;++j)
                 std::cout<<yi[offset+j]<<(j==nb_comp-1?" ]":" ");
               std::cout<<std::endl ;
               offset += nb_comp ;
             }
         }
       }
    }

    if(vm["test-carnot"].as<int>() > 0 )
    {
       std::string data_path = vm["data-file"].as<std::string>();
       int nb_comp = vm["nb-comp"].as<int>() ;
       int test_data_id = vm["test-data-id"].as<int>() ;
       int batch_size   = vm["batch-size"].as<int>() ;

       {
         auto data = cnpy::npy_load(data_path) ;
         auto data_dims = data.shape.size() ;
         assert(data_dims == 2) ;

         auto nrows = data.shape[0] ;
         auto ncols = data.shape[1] ;

         std::cout<<"nb rows  ="<<nrows<<std::endl ;
         std::cout<<"nb cols  ="<<ncols<<std::endl ;
         std::cout<<"nb comps ="<<nb_comp<<std::endl ;

         assert(ncols >= 2 + nb_comp) ;

         typedef double value_type ;
         std::size_t tensor_size = 2 + nb_comp ;
         std::vector<double> x(batch_size*tensor_size) ;
         std::cout<<"TEST DATA ID : "<<test_data_id<<std::endl ;
         int offset = 0 ;
         for(int i=0;i<batch_size;++i)
         {
             int id = (test_data_id+i)%nrows ;
             std::cout<<"[P, T, Z ] : [";
             for(int ic=0;ic<tensor_size;++ic)
             {
                x[offset+ic] = data.data<value_type>()[ic*nrows+id] ;
                std::cout<<x[offset + ic]<<",";
             }
             std::cout<<"]"<<std::endl ;
             offset += tensor_size ;
         }
       }


       std::vector<bool>     unstable ;
       std::vector<double>   theta_l ;
       std::vector<double>   vol_l ;
       std::vector<double>   theta_v ;
       std::vector<double>   vol_v ;
       std::vector<double>   xi ;
       std::vector<double>   yi ;
       std::vector<double>   xki ;
       std::vector<double>   yki ;
       {
         std::string data_path = vm["result-file"].as<std::string>();
         auto data = cnpy::npy_load(data_path) ;
         auto data_dims = data.shape.size() ;
         assert(data_dims == 2) ;

         auto nrows = data.shape[0] ;
         auto ncols = data.shape[1] ;
         assert(ncols == 6*nb_comp+6) ;

         std::cout<<"nb rows  = "<<nrows<<std::endl ;
         std::cout<<"nb cols  = "<<ncols<<std::endl ;

         theta_l.resize(batch_size) ;
         vol_l.resize(batch_size) ;
         theta_v.resize(batch_size) ;
         vol_v.resize(batch_size) ;
         xi.resize(batch_size*nb_comp) ;
         yi.resize(batch_size*nb_comp) ;
         xki.resize(batch_size*nb_comp) ;
         yki.resize(batch_size*nb_comp) ;
         typedef double value_type ;
         int offset = 0 ;
         for(int i=0;i<batch_size;++i)
         {
             int id = (test_data_id+i)%nrows ;
             theta_l[i] = data.data<value_type>()[id*ncols+nb_comp];
             vol_l[i] = data.data<value_type>()[id*ncols+2*nb_comp+1];
             theta_v[i] = data.data<value_type>()[id*ncols+3*nb_comp+2];
             vol_v[i] = data.data<value_type>()[id*ncols+4*nb_comp+3];
             for(int ic=0;ic<nb_comp;++ic)
             {
                xi[offset+ic] = data.data<value_type>()[id*ncols+ic];
                xki[offset+ic] = data.data<value_type>()[id*ncols+nb_comp+1+ic];
                yi[offset+ic] = data.data<value_type>()[id*ncols+2*nb_comp+2+ic];
                yki[offset+ic] = data.data<value_type>()[id*ncols+3*nb_comp+3+ic];
             }
             /*
             std::cout<<"RES["<<i<<"]:(";
             for(int j=0;j<4*nb_comp+2;++j)
             {
                 std::cout<<data.data<value_type>()[id*ncols+j]<<",";
             }
             std::cout<<std::endl ;
             */

             offset += nb_comp ;
         }
         for(int i=0;i<batch_size;++i)
         {
             std::cout<<"THETA (L,V) ["<<i<<"]: ("<<theta_l[i]<<","<<theta_v[i]<<")"<<std::endl ;
         }
         offset = 0 ;
         for(int i=0;i<batch_size;++i)
         {
             std::cout<<"XI ["<<i<<"]:";
             for(int j=0;j<nb_comp;++j)
                std::cout<<xi[offset+j]<<",";
             std::cout<<std::endl ;
             std::cout<<"YI ["<<i<<"]:";
             for(int j=0;j<nb_comp;++j)
                std::cout<<yi[offset+j]<<",";
             std::cout<<std::endl ;
             std::cout<<"XKI ["<<i<<"]:";
             for(int j=0;j<nb_comp;++j)
                std::cout<<xki[offset+j]<<",";
             std::cout<<std::endl ;
             std::cout<<"YKI ["<<i<<"]:";
             for(int j=0;j<nb_comp;++j)
                std::cout<<yki[offset+j]<<",";
             std::cout<<std::endl ;
             offset += nb_comp ;
         }

       }
    }



    perf_mng.printInfo();


    return 0;
}
