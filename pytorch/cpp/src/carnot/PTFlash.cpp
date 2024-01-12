/*
 * PTFlash.cc
 *
 *  Created on: 3 août 2023
 *      Author: gratienj
 */

#include <cassert>
#include <cmath>
#include <exception>
#include <iomanip>

#ifdef USE_CARNOT
#include "IComponent.h"
#include "IFluid.h"
//#include "IEquilibriumResult.h"
#include "ISystem.h"
#include "eEquilibriumModelType.h"

using namespace carnot ;
//#define USE_CARNOT_V9
#define USE_CARNOT_V10
#endif

#ifdef USE_TORCH
#include <torch/script.h>
#include <torch/serialize/tensor.h>
#include <torch/serialize.h>
#endif

#ifdef USE_ONNX
//#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime_cxx_api.h>
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

#include "tensorrt/TRTEngine.h"
#endif

#include "utils/PerfCounterMng.h"
#include "PTFlash.h"


struct PTFlash::CarnotInternal
{

  typedef enum {
    Aqueous,
    Liquid,
    Vapour
  } ePhaseType ;

  static const int nbPhases = 2 ;
#ifdef USE_CARNOT

  void createArea(std::string const& bank_path) {
#ifdef USE_CARNOT_V9
   /*
   m_area = IThermoAreaFactory::create(eEquilibriumModel::SoaveRedlichKwong,
                                       ePhaseModel::SoaveRedlichKwong,
                                       eViscoModel::LBC);
                                       */
   m_area = IThermoAreaFactory::create(eEquilibriumModel::SoaveRedlichKwong,
                                       ePhaseModel::DefaultPhaseModel,
                                       eViscoModel::DefaultViscoModel);
   IComponentBank::loadBank(bank_path) ;
#endif
#ifdef USE_CARNOT_V10
   /*
    std::vector<std::string> configs =  IThermoAreaFactory::getAvailableConfigs() ;
    std::cout<<"AVAILABLE CONFIGS : ";
    for( auto const& c :  configs)
      std::cout<<c<<",";
    std::cout<<std::endl ;
    */
    m_area = IThermoAreaFactory::create("SRK") ;

    m_area->loadBank(bank_path) ;
    /*
    const std::vector<int> uids = m_area->getAvailableCAS() ;
    std::cout<<"AVAILABLE CAS : ";
    for( auto const& uid :  uids)
      std::cout<<uid<<",";
    std::cout<<std::endl ;
    */
#endif

  }

  void createFluid()
  {
    m_pfluid = IFluidFactory::create(m_area.get());
    //IThermoModel& thermoModel = m_pfluid->getThermoModel();
    //thermoModel.setModels(eEquilibriumModel::SoaveRedlichKwong,ePhaseModel::SoaveRedlichKwong, eViscoModel::LBC);
  }

  void addComponent(int uid)
  {
     //auto comp = carnot::IComponentFactory::create(uid);
     //IComponent* comp(m_area->addComponent(uid));
     //comp->computeAll();
    try {
#ifdef USE_CARNOT_V10
     auto comp = m_area->createComponentFromDataBank(uid) ;
     m_pfluid->addComponent(comp);
     auto current_id = m_components.size() ;
     m_components.push_back(comp.get()) ;
     m_component_ids.push_back(m_current_id) ;
#endif
#ifdef USE_CARNOT_V9
     IComponent* comp(m_area->addComponent(uid));
     m_pfluid->addComponent(comp);
     auto current_id = m_components.size() ;
     m_components.push_back(comp) ;
     m_component_ids.push_back(m_current_id) ;
#endif
    }
    catch(std::exception exc)
    {
        std::cerr<<"ERROR WHILE CREATING COMPONENT UID : "<<uid<<" msg:"<<exc.what();
    }
  }

  std::size_t nbComponents() const {
    return m_components.size() ;
  }

  void setComposition(double const* zi) {
    std::vector<double> composition{zi,zi+nbComponents()} ;
     m_pfluid->setComposition(composition);
  }

  void setPropertyAsDouble(eProperty::Type property,double value){
    m_pfluid->setPropertyAsDouble(property, value);
  }


  double getPropertyAsDouble(eProperty::Type property){
    return m_pfluid->getPropertyAsDouble(property);
  }

  //std::shared_ptr<IEquilibriumResult> computeEquilibrium(eEquilibrium::Type equilibrium_type)
  std::shared_ptr<ISystem> computeEquilibrium(eEquilibrium::Type equilibrium_type)
  {
    return m_pfluid->computeEquilibrium(equilibrium_type) ;
  }

  int componentUid(std::string const& component_name) {
     auto iter = m_carnot_uids.find(component_name) ;
     if(iter == m_carnot_uids.end())
       return -1 ;
     else
       return iter->second ;
  }

  int componentId(int ic) {
    assert( (ic>=0)&& (ic<m_component_ids.size())) ;
    return m_component_ids[ic] ;
  }

  std::string componentName(int uid) {
    return m_carnot_names[uid] ;
  }

private :
  std::map<std::string,int> m_carnot_uids = {
      { "H2O", 7732185 },
      { "H2S", 7783064 },
      { "CO2", 124389  },
      { "CH4", 74828   },
      { "C2H6", 74840  },
      { "C3H8", 74986  },
      { "C4H10", 106978},
      { "C5H12", 109660},
      { "C6H14", 110543},
      { "C7H16", 142825},
      { "CH3OH", 67561 },
      { "N2",  7727379 },
      { "O2",  7782447 }} ;

  std::map<int,std::string> m_carnot_names = {
      { 7732185, "H2O" },
      { 7783064, "H2S" },
      { 124389,  "CO2" },
      { 74828,   "CH4" },
      { 67561,   "CH3OH"},
      { 7727379, "N2"   },
      { 7782447, "O2"   },
      { 74840,   "C2H6" },
      { 74986,   "C3H8" },
      { 106978,  "C4H10"},
      { 109660,  "C5H12"},
      { 110543,  "C6H14"},
      { 142825,  "C7H16"}
  } ;

  std::unique_ptr<IThermoArea>             m_area;
  std::shared_ptr<IFluid>                  m_pfluid ;
  std::vector<IComponent*>                 m_components ;
  std::vector<int>                         m_component_ids ;
  int                                      m_current_id = 0 ;
#endif
} ;

void
PTFlash::initCarnot(std::string const& carnot_bank_path,
                   int num_phase,
                   int num_compo,
                   std::vector<int> const& component_uids)
{
  m_num_phase = num_phase ;
  m_num_compo = num_compo ;
  m_carnot_internal = new CarnotInternal() ;
  m_use_carnot = true ;

#ifdef USE_CARNOT

  {
     std::cout<< " LOAD CARNOT BANK : "<<carnot_bank_path<<std::endl ;
     //IComponentFactory::loadBank(carnot_bank_path.localstr());
     //IComponentBank::loadBank(carnot_model_path.localstr());
  }

  // SETTING THERMODYNAMIC MODEL
  ////////////////////////////////////////////////////////////////////
  // Avaliable models : CPA_Converge, PR_Convege, BWRS_Converge
  ////////////////////////////////////////////////////////////////////
  m_carnot_internal->createArea(carnot_bank_path);

  // fluid declaration
  m_carnot_internal->createFluid();

  for(int ic=0;ic<m_num_compo;++ic)
  {
    //auto comp_uid = carnot_model->componentUid(comp_name) ;
    //int comp_uid = m_carnot_internal->componentUid(component_uids[ic]) ;
    //assert(comp_uid != -1) ;
    //m_carnot_internal->addComponent(comp_uid) ;
    m_carnot_internal->addComponent(component_uids[ic]) ;
  }
#endif
  m_perf_mng.init("Carnot::Init") ;
  m_perf_mng.init("Carnot::Prepare") ;
  m_perf_mng.init("Carnot::Compute") ;
  m_perf_mng.init("Carnot::End") ;
}

void PTFlash::_endComputeCarnot() const
{
#ifdef USE_CARNOT
  m_perf_mng.start("Carnot::Prepare") ;
  auto nb_components       = m_carnot_internal->nbComponents() ;
  std::size_t offset = 0 ;
  std::size_t unstable_offset = 0 ;
  m_current_index = 0 ;
  m_current_unstable_index = 0 ;
  m_unstable.resize(m_batch_size) ;
  m_theta_v.resize(m_batch_size) ;
  m_xi.resize(m_batch_size*nb_components) ;
  m_yi.resize(m_batch_size*nb_components) ;
  m_ki.resize(m_batch_size*nb_components) ;
  m_perf_mng.stop("Carnot::Prepare") ;

  for(std::size_t i=0;i<m_batch_size;++i)
  {
   m_perf_mng.start("Carnot::Prepare") ;
   // set pressure
   m_carnot_internal->setPropertyAsDouble(eProperty::Pressure, m_x[offset]);

   // set  temperature
   m_carnot_internal->setPropertyAsDouble(eProperty::Temperature, m_x[offset+1]);

   // set composition on the fluid
   m_carnot_internal->setComposition(&m_x[offset+2]);
   m_perf_mng.stop("Carnot::Prepare") ;
   //
   // COMPUTE PT flash


   if(m_output_level>0)
   {
     std::cout<<i<<" [P, T, Z ] : "<<std::setprecision(16)<<m_x[offset]<<" "<< m_x[offset+1]<<" [";
     for(int ic=0;ic<nb_components;++ic)
      std::cout<<m_x[offset+2+ic]<< (ic==nb_components-1?"]":" ");
     std::cout<<std::endl ;
   }

   m_perf_mng.start("Carnot::Compute") ;
   auto res = m_carnot_internal->computeEquilibrium(eEquilibrium::PT);
   m_perf_mng.stop("Carnot::Compute") ;

   m_perf_mng.start("Carnot::End") ;
   double theta    = NAN ;
   double thetaLiq = NAN ;
   double thetaVap = NAN ;
   //unsigned int nb_phases = 0 ;

   if (res.get() != nullptr)
   {
      //theta = res->getTheta();
      theta = res->getPropertyAsDouble(eProperty::Theta) ;
    }

   m_unstable[m_current_index] = false ;

   auto allFluids = res->getFluids();

   if (theta > 0.0 && theta < 1.)
   { // two-phase fluid
     m_unstable[m_current_index] = true ;
     m_theta_v[m_current_unstable_index] =  theta ;

     //auto fluidLiq = IEquilibriumResult::getSharedLiquidOutput(res);
     //auto fluidVap = IEquilibriumResult::getSharedVaporOutput(res);
     //std::vector<double> compoLiq = fluidLiq->getComposition();
     //std::vector<double> compoVap = fluidVap->getComposition();
#ifdef USE_CARNOT_V9
     try {
        auto fluidLiq = res->getFluid(eFluidState::Liquid);
        thetaLiq = fluidLiq->getNbMoles() ;

        std::vector<double> compoLiq = fluidLiq->getComposition();
        for(int ic=0;ic<nb_components;++ic)
         {
           m_xi[unstable_offset+ic] = compoLiq[ic] ;
         }
     }
     catch(std::exception exc)
     {
        std::cout<<"LIQUID EXCEPTION :"<<exc.what()<<std::endl ;
     }

     try {
        auto fluidVap = res->getFluid(eFluidState::Vapour);
        thetaVap = fluidVap->getNbMoles() ;
        std::vector<double> compoVap = fluidVap->getComposition();
        for(int ic=0;ic<nb_components;++ic)
         {
           m_yi[unstable_offset+ic] = compoVap[ic] ;
         }
     }
     catch(std::exception exc)
     {
        std::cout<<"VAP EXCEPTION :"<<exc.what()<<std::endl ;
     }
#endif
#ifdef USE_CARNOT_V10
     //auto fluidLiq = res->getFluid(eFluidState::Liquid);
     //auto fluidVap = res->getFluid(eFluidState::Vapour);
     //auto fluidLiq = allFluids[carnot::eFluidState::Liquid];
     //auto fluidVap = allFluids[carnot::eFluidState::Vapour];

     assert(allFluids.size()==2) ;
     auto fluidLiq = allFluids[0]->getState() == carnot::eFluidState::Liquid?allFluids[0]:allFluids[1] ;
     auto fluidVap = allFluids[0]->getState() == carnot::eFluidState::Liquid?allFluids[1]:allFluids[0] ;

     thetaLiq = fluidLiq->getNbMoles() ;
     thetaVap = fluidVap->getNbMoles() ;

     std::vector<double> compoLiq = fluidLiq->getComposition();
     std::vector<double> compoVap = fluidVap->getComposition();
     for(int ic=0;ic<nb_components;++ic)
     {
      m_xi[unstable_offset+ic] = compoLiq[ic] ;
      m_yi[unstable_offset+ic] = compoVap[ic] ;
     }
#endif
     ++m_current_unstable_index ;
     unstable_offset += nb_components ;
   }
   else
   {
     if (theta == 1.)
     {
      //auto fluidVap = res->getFluid(eFluidState::Vapour);
      //auto fluidVap = allFluids[carnot::eFluidState::Vapour];
      assert(allFluids.size()==1) ;
      auto fluidVap = allFluids[0];
      thetaVap = fluidVap->getNbMoles() ;
      std::vector<double> compoVap = fluidVap->getComposition();
      for(int ic=0;ic<nb_components;++ic)
      {
        m_yi[unstable_offset+ic] = compoVap[ic] ;
      }
     }
     if (theta == 0.)
     {
      //auto fluidLiq = res->getFluid(eFluidState::Liquid);
      //auto fluidLiq = allFluids[carnot::eFluidState::Liquid];
      auto fluidLiq = allFluids[0];
      thetaLiq = fluidLiq->getNbMoles() ;
      std::vector<double> compoLiq = fluidLiq->getComposition();
      for(int ic=0;ic<nb_components;++ic)
      {
        m_xi[unstable_offset+ic] = compoLiq[ic] ;
      }
     }
   }
   if(m_output_level>0)
   {
     std::cout<<"THETA    ["<<i<<"]="<<theta<<" "<<thetaLiq<<" "<<thetaVap<<std::endl ;
     std::cout<<"THETA LIQ["<<i<<"]="<<thetaLiq<<std::endl ;
     std::cout<<"THETA VAP["<<i<<"]="<<thetaVap<<std::endl ;
   }
   m_perf_mng.stop("Carnot::End") ;

   offset += m_tensor_size ;
   ++ m_current_index ;
  }
#endif
}

void
PTFlash::initCAWF(std::string const& ptflash_model_path,
                  int num_phase,
                  int num_compo)
{
  m_num_phase = num_phase ;
  m_num_compo = num_compo ;

#ifdef USE_CAWFINFERENCE
  m_cawf_inference_mng.init("localhost",50051) ;
  std::cout<<"CAWF INFERENCE SERVER CONNECTED:"<<std::endl ;
  std::cout<<"CAWF INFERENCE SERVER LOAD MODEL :"<<ptflash_model_path<<std::endl ;
  m_cawf_inference_mng.loadModel(ptflash_model_path) ;
  std::cout<<"CAWF INFERENCE MODEL : "<<ptflash_model_path<<" LOADED:"<<std::endl ;
  m_use_cawf_inference = true ;
#endif

  m_perf_mng.init("CAWF::Init") ;
  m_perf_mng.init("CAWF::Prepare") ;
  m_perf_mng.init("CAWF::Compute") ;
  m_perf_mng.init("CAWF::End") ;
}

void PTFlash::_endComputeCAWF() const
{
#ifdef USE_CAWFINFERENCE
    m_perf_mng.start("CAWF::Prepare") ;
    cawf_inference::CAWFInferenceMng::Input input;
    int dims[2] = { m_batch_size, m_tensor_size } ;
    input.setDoubleTensorDims(2,dims) ;
    input.addDoubleBufferValues(m_x.data(),m_x.size()) ;
    cawf_inference::CAWFInferenceMng::Output output;
    m_perf_mng.stop("CAWF::Prepare") ;


    m_perf_mng.start("CAWF::Compute") ;
    m_cawf_inference_mng.evalDNN(input,output) ;
    m_perf_mng.stop("CAWF::Compute") ;


    m_perf_mng.start("CAWF::End") ;
    {
      assert(output.nbBoolBuffer()==1) ;
      {
        int ndim = output.getBoolTensorNbDims(m_unstable_id) ;
        assert(ndim == 1) ;
        std::vector<int> dims(ndim) ;
        output.getBoolTensorDims(m_unstable_id,dims.data(),ndim) ;
        assert(dims[0] == m_batch_size) ;
        assert(output.getBoolBufferSize(m_unstable_id)==m_batch_size) ;
        m_unstable.resize(dims[0]) ;
        output.getBoolBufferValues(m_unstable_id,m_unstable,m_batch_size) ;
      }

      assert(output.nbDoubleBuffer()==4) ;
      {
        std::size_t nb_unstable_compo = 0 ;
        {
          int ndim = output.getDoubleTensorNbDims(m_theta_v_id) ;
          assert(ndim == 2) ;
          std::vector<int> dims(ndim) ;
          output.getDoubleTensorDims(m_theta_v_id,dims.data(),ndim) ;
          nb_unstable_compo = dims[0] ;
          assert(dims[1] == 1) ;
          m_theta_v.resize(nb_unstable_compo) ;
          assert(output.getDoubleBufferSize(m_theta_v_id) == nb_unstable_compo) ;
          output.getDoubleBufferValues(m_theta_v_id,m_theta_v.data(),nb_unstable_compo) ;
        }
        {
          int ndim = output.getDoubleTensorNbDims(m_xi_id) ;
          assert(ndim == 2) ;
          std::vector<int> dims(ndim) ;
          output.getDoubleTensorDims(m_xi_id,dims.data(),2) ;
          assert(nb_unstable_compo == dims[0]) ;
          std::size_t tensor_size = dims[0]*dims[1] ;
          m_xi.resize(tensor_size) ;
          assert(output.getDoubleBufferSize(m_xi_id) == tensor_size) ;
          output.getDoubleBufferValues(m_xi_id,m_xi.data(),tensor_size) ;
        }
        {
          int ndim = output.getDoubleTensorNbDims(m_yi_id) ;
          assert(ndim == 2) ;
          std::vector<int> dims(2) ;
          output.getDoubleTensorDims(m_yi_id,dims.data(),2) ;
          assert(nb_unstable_compo == dims[0]) ;
          std::size_t tensor_size = dims[0]*dims[1] ;
          m_yi.resize(tensor_size) ;
          assert(output.getDoubleBufferSize(m_yi_id) == tensor_size) ;
          output.getDoubleBufferValues(m_yi_id,m_yi.data(),tensor_size) ;
        }
        {
          int ndim = output.getDoubleTensorNbDims(m_ki_id) ;
          assert(ndim == 2) ;
          std::vector<int> dims(2) ;
          output.getDoubleTensorDims(m_ki_id,dims.data(),2) ;
          assert(nb_unstable_compo == dims[0]) ;
          std::size_t tensor_size = dims[0]*dims[1] ;
          m_ki.resize(tensor_size) ;
          output.getDoubleBufferValues(m_ki_id,m_ki.data(),tensor_size) ;

        }
      }
    }
    m_perf_mng.stop("CAWF::Compute") ;
#endif
}

struct PTFlash::TorchInternal
{
#ifdef USE_TORCH
  torch::jit::script::Module m_module;
  torch::jit::script::Module m_classifier_module;
  torch::jit::script::Module m_initializer_module;
  torch::Tensor m_x ;
  torch::Tensor m_x2 ;
#endif
} ;

void
PTFlash::initTorch(std::string const& ptflash_model_path,
                   int num_phase,
                   int num_compo,
                   bool use_gpu)
{
  m_use_gpu = use_gpu ;
  m_num_phase = num_phase ;
  m_num_compo = num_compo ;
  m_flash_model_path = ptflash_model_path ;
  m_model_type = FullFlash ;
  m_use_torch = true ;
  {
    m_torch_internal = new TorchInternal() ;
#ifdef USE_TORCH
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
        std::cout<<"TRY TO LOAD PTFLASH MODEL : "<<ptflash_model_path<<std::endl ;
        m_torch_internal->m_module = torch::jit::load(ptflash_model_path);
        std::cout<<"Torch PTFlash module is loaded"<<std::endl ;
    }
    catch (const c10::Error& e) {
      std::cerr << "error loading the model\n"<<e.msg();
    }
#endif
  }
  m_perf_mng.init("Torch::Init") ;
  m_perf_mng.init("Torch::Prepare") ;
  m_perf_mng.init("Torch::Compute") ;
  m_perf_mng.init("Torch::End") ;
}

void
PTFlash::initTorch(std::string const& classifier_model_path,
                   std::string const& initializer_model_path,
                   eModelDNNType model,
                   int num_phase,
                   int num_compo,
                   bool use_gpu,
                   bool use_fp32)
{
  m_classifier_model_path = classifier_model_path ;
  m_initializer_model_path = initializer_model_path ;
  m_model_type = model ;
  m_num_phase = num_phase ;
  m_num_compo = num_compo ;
  m_use_gpu = use_gpu ;
  m_use_fp32 = use_fp32 ;
  m_use_torch = true ;

  m_torch_internal = new TorchInternal() ;
  switch(m_model_type)
  {
  case Classifier :
#ifdef USE_TORCH
    try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
      std::cout<<"TRY TO LOAD CLASSIFIER MODEL : "<<classifier_model_path<<std::endl ;
      m_torch_internal->m_classifier_module = torch::jit::load(classifier_model_path);

      torch::DeviceType device_type = m_use_gpu ? torch::kCUDA : torch::kCPU;
      torch::Device device(device_type);
      m_torch_internal->m_classifier_module.to(device);

      std::cout<<"Torch Classifier module is loaded"<<std::endl ;
    }
    catch (const c10::Error& e) {
      std::cerr << "error loading the model\n"<<e.msg();
    }
#endif
    break ;
  case Initializer :
#ifdef USE_TORCH
    try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
      std::cout<<"TRY TO LOAD CLASSIFIER MODEL : "<<initializer_model_path<<std::endl ;
      m_torch_internal->m_initializer_module = torch::jit::load(initializer_model_path);

      torch::DeviceType device_type = m_use_gpu ? torch::kCUDA : torch::kCPU;
      torch::Device device(device_type);
      m_torch_internal->m_initializer_module.to(device);

      std::cout<<"Torch Initializer module is loaded"<<std::endl ;
    }
    catch (const c10::Error& e) {
      std::cerr << "error loading the model\n"<<e.msg();
    }
#endif
    break ;
  case FullClassInit :
#ifdef USE_TORCH
    try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
      std::cout<<"TRY TO LOAD CLASSIFIER MODEL : "<<classifier_model_path<<std::endl ;
      m_torch_internal->m_classifier_module = torch::jit::load(classifier_model_path);

      torch::DeviceType device_type = m_use_gpu ? torch::kCUDA : torch::kCPU;
      torch::Device device(device_type);
      m_torch_internal->m_classifier_module.to(device);

      std::cout<<"Torch Classifier module is loaded"<<std::endl ;
    }
    catch (const c10::Error& e) {
      std::cerr << "error loading the model\n"<<e.msg();
    }
    try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
      std::cout<<"TRY TO LOAD CLASSIFIER MODEL : "<<initializer_model_path<<std::endl ;
      m_torch_internal->m_initializer_module = torch::jit::load(initializer_model_path);

      torch::DeviceType device_type = m_use_gpu ? torch::kCUDA : torch::kCPU;
      torch::Device device(device_type);
      m_torch_internal->m_initializer_module.to(device);
      std::cout<<"Torch Initializer module is loaded"<<std::endl ;
    }
    catch (const c10::Error& e) {
      std::cerr << "error loading the model\n"<<e.msg();
    }
#endif
    break ;
  default:
    break ;
  }
  m_perf_mng.init("Torch::Init") ;
  m_perf_mng.init("Torch::Prepare") ;
  m_perf_mng.init("Torch::Compute") ;
  m_perf_mng.init("Torch::End") ;
}


void PTFlash::_endComputeTorch() const
{
#ifdef USE_TORCH
    m_perf_mng.start("Torch::Prepare") ;
    std::vector<int64_t> dims = { m_batch_size, m_tensor_size};
    std::vector<torch::jit::IValue> inputs;
    torch::DeviceType device_type = m_use_gpu ? torch::kCUDA : torch::kCPU;
    torch::Device device(device_type);

    if(m_use_fp32)
    {
      torch::TensorOptions options(torch::kFloat32);
      m_torch_internal->m_x = torch::from_blob(m_xf.data(), torch::IntList(dims), options).clone();
    }
    else
    {
      torch::TensorOptions options(torch::kFloat64);
      m_torch_internal->m_x = torch::from_blob(m_x.data(), torch::IntList(dims), options).clone();
    }
    m_torch_internal->m_x = m_torch_internal->m_x.to(device) ;
    inputs.push_back(m_torch_internal->m_x);
    torch::NoGradGuard no_grad;
    m_perf_mng.stop("Torch::Prepare") ;
    switch(m_model_type)
    {
    case FullFlash :
    {
      std::cout<<" TORCH FullFlash FORWARD"<<std::endl ;
      m_perf_mng.start("Torch::Compute") ;
      auto outputs = m_torch_internal->m_module.forward(inputs).toTuple();
      m_perf_mng.stop("Torch::Compute") ;

      m_perf_mng.start("Torch::End") ;
      torch::Device cpu_device(torch::kCPU);
      {
          auto out = outputs->elements()[0].toTensor().to(cpu_device);
          torch::ArrayRef<int64_t> sizes = out.sizes();
          std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes.size()<<std::endl ;
          assert(sizes.size()==1) ;
          assert(sizes[0]==m_batch_size) ;
          m_unstable.resize(m_batch_size) ;
          m_unstable.assign(out.data_ptr<bool>(),out.data_ptr<bool>() + sizes[0]) ;
      }
      {
          auto out = outputs->elements()[1].toTensor().to(cpu_device);
          torch::ArrayRef<int64_t> sizes = out.sizes();
          std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes[1]<<" "<<sizes.size()<<std::endl ;
          assert(sizes.size()==2) ;
          assert(sizes[0]<=m_batch_size) ;
          assert(sizes[1]==1) ;
          if(sizes[0]>0)
          {
              m_theta_v.resize(sizes[0]) ;
              m_theta_v.assign(out.data_ptr<double>(),out.data_ptr<double>() + sizes[0]*sizes[1]);
          }
      }
      {
          auto out = outputs->elements()[2].toTensor().to(cpu_device);
          torch::ArrayRef<int64_t> sizes = out.sizes();
          std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes[1]<<" "<<sizes.size()<<std::endl ;
          assert(sizes.size()==2) ;
          assert(sizes[0]<=m_batch_size) ;
          assert(sizes[1]==m_num_compo) ;
          if(sizes[0]>0)
          {
              m_xi.resize(sizes[0]*sizes[1]) ;
              m_xi.assign(out.data_ptr<double>(),out.data_ptr<double>() + sizes[0]*sizes[1]);
          }
      }
      {
          auto out = outputs->elements()[3].toTensor().to(cpu_device);
          torch::ArrayRef<int64_t> sizes = out.sizes();
          std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes[1]<<" "<<sizes.size()<<std::endl ;
          assert(sizes.size()==2) ;
          assert(sizes[0]<=m_batch_size) ;
          assert(sizes[1]==m_num_compo) ;
          if(sizes[0]>0)
          {
              m_yi.resize(sizes[0]*sizes[1]) ;
              m_yi.assign(out.data_ptr<double>(),out.data_ptr<double>() + sizes[0]*sizes[1]);
          }
      }
      {
          auto out = outputs->elements()[4].toTensor().to(cpu_device);
          torch::ArrayRef<int64_t> sizes = out.sizes();
          std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes[1]<<" "<<sizes.size()<<std::endl ;
          assert(sizes.size()==2) ;
          assert(sizes[0]<=m_batch_size) ;
          assert(sizes[1]==m_num_compo) ;
          if(sizes[0]>0)
          {
              m_ki.resize(sizes[0]*sizes[1]) ;
              m_ki.assign(out.data_ptr<double>(),out.data_ptr<double>() + sizes[0]*sizes[1]);
          }
      }
      m_perf_mng.stop("Torch::End") ;
    }
    break ;
    case Classifier:
    {
      std::cout<<"TORCH CLASSIFIER FORWARD"<<std::endl ;
      m_perf_mng.start("Torch::Compute") ;
      auto outputs = m_torch_internal->m_classifier_module.forward(inputs);
      m_perf_mng.stop("Torch::Compute") ;

      m_perf_mng.start("Torch::End") ;

      m_current_index = 0 ;
      m_current_unstable_index = 0 ;
      m_unstable.resize(m_batch_size) ;
      m_theta_v.resize(m_batch_size) ;
      m_xi.resize(m_batch_size*m_num_compo) ;
      m_yi.resize(m_batch_size*m_num_compo) ;
      m_ki.resize(m_batch_size*m_num_compo) ;

      torch::Device cpu_device(torch::kCPU);
      {
          auto out = outputs.toTensor().to(cpu_device);
          torch::ArrayRef<int64_t> sizes = out.sizes();
          std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes.size()<<std::endl ;
          assert(sizes.size()==2) ;
          assert(sizes[0]==m_batch_size) ;
          assert(sizes[1]==1) ;
          if(m_use_fp32)
          {
            float* y = out.data_ptr<float>() ;
            for(int i=0;i<m_batch_size;++i)
            {
              float prob = 1./(1+std::exp(-y[i])) ;
              m_unstable[i] = prob > 0.5 ;
            }
          }
          else
          {
            double* y = out.data_ptr<double>() ;
            for(int i=0;i<m_batch_size;++i)
            {
              double prob = 1./(1+std::exp(-y[i])) ;
              m_unstable[i] = prob > 0.5 ;
            }
          }
      }
      m_perf_mng.stop("Torch::End") ;
    }
    break ;
    case FullClassInit:
    {
      std::cout<<"TORCH CLASSIFIER FORWARD"<<std::endl ;
      m_perf_mng.start("Torch::Compute") ;
      auto outputs = m_torch_internal->m_classifier_module.forward(inputs);
      m_perf_mng.stop("Torch::Compute") ;

      m_current_index = 0 ;
      m_current_unstable_index = 0 ;
      m_unstable.resize(m_batch_size) ;
      m_theta_v.resize(m_batch_size) ;
      m_xi.resize(m_batch_size*m_num_compo) ;
      m_yi.resize(m_batch_size*m_num_compo) ;
      m_ki.resize(m_batch_size*m_num_compo) ;

      torch::Device cpu_device(torch::kCPU);
      {
          m_perf_mng.start("Torch::End") ;
          auto out = outputs.toTensor().to(cpu_device);
          torch::ArrayRef<int64_t> sizes = out.sizes();
          std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes.size()<<std::endl ;
          assert(sizes.size()==2) ;
          assert(sizes[0]==m_batch_size) ;
          assert(sizes[1]==1) ;

          if(m_use_fp32)
          {
            float* y = out.data_ptr<float>() ;
            for(int i=0;i<m_batch_size;++i)
            {
              float prob = 1./(1+std::exp(-y[i])) ;
              m_unstable[i] = prob > 0.5 ;
            }
          }
          else
          {
            double* y = out.data_ptr<double>() ;
            for(int i=0;i<m_batch_size;++i)
            {
              double prob = 1./(1+std::exp(-y[i])) ;
              m_unstable[i] = prob > 0.5 ;
            }
          }
          int offset = 0 ;
          int unstable_offset = 0 ;
          int nb_unstable = 0 ;

          if(m_use_fp32)
          {
            m_x2f.resize(0) ;
            m_x2f.reserve(m_batch_size) ;
            for(std::size_t i=0;i<m_batch_size;++i)
            {
                if(m_unstable[i])
                {
                    for(int j=0;j<m_tensor_size;++j)
                      m_x2f.push_back(m_xf[offset+j]) ;
                    unstable_offset += m_tensor_size ;
                    ++ nb_unstable ;
                }
                offset += m_tensor_size ;
            }
            assert(m_x2f.size()==unstable_offset) ;
          }
          else
          {
            m_x2.resize(0) ;
            m_x2.reserve(m_batch_size) ;
            for(std::size_t i=0;i<m_batch_size;++i)
            {
                if(m_unstable[i])
                {
                    for(int j=0;j<m_tensor_size;++j)
                      m_x2.push_back(m_x[offset+j]) ;
                    unstable_offset += m_tensor_size ;
                    ++ nb_unstable ;
                }
                offset += m_tensor_size ;
            }
            assert(m_x2.size()==unstable_offset) ;
          }
          m_perf_mng.stop("Torch::End") ;

          if(nb_unstable>0)
          {
            //torch::Device cuda_device(torch::kCUDA);
            std::vector<int64_t> initializer_dims = { nb_unstable, m_tensor_size};
            std::vector<torch::jit::IValue> initializer_inputs;
            if(m_use_fp32)
            {
              torch::TensorOptions options(torch::kFloat32);
              m_torch_internal->m_x2 = torch::from_blob(m_x2f.data(), torch::IntList(initializer_dims), options).clone();
            }
            else
            {
              torch::TensorOptions options(torch::kFloat64);
              m_torch_internal->m_x2 = torch::from_blob(m_x2.data(), torch::IntList(initializer_dims), options).clone();
            }

            m_torch_internal->m_x2 = m_torch_internal->m_x2.to(device) ;
            initializer_inputs.push_back(m_torch_internal->m_x2);

            std::cout<<"TORCH INITIALIZER FORWARD"<<std::endl ;
            m_perf_mng.start("Torch::Compute") ;
            auto outputs = m_torch_internal->m_initializer_module.forward(initializer_inputs);
            m_perf_mng.stop("Torch::Compute") ;

            m_perf_mng.start("Torch::End") ;
            torch::Device cpu_device(torch::kCPU);
            {
                auto out = outputs.toTensor().to(cpu_device);
                torch::ArrayRef<int64_t> sizes = out.sizes();
                std::cout<<"SIZES : "<<sizes[0]<<" "<<sizes.size()<<std::endl ;
                assert(sizes.size()==2) ;
                assert(sizes[0]==nb_unstable) ;
                assert(sizes[1]==m_num_compo) ;
                m_ki.resize(nb_unstable*m_num_compo) ;
                if(m_use_fp32)
                  m_ki.assign(out.data_ptr<float>(),out.data_ptr<float>() + sizes[0]*sizes[1]) ;
                else
                  m_ki.assign(out.data_ptr<double>(),out.data_ptr<double>() + sizes[0]*sizes[1]) ;
            }
            m_perf_mng.stop("Torch::End") ;
          }
      }
    }
    break ;
    default :
    break ;
  }
#endif
}

struct PTFlash::ONNXInternal
{
  ONNXInternal()
#ifdef USE_ONNX
: m_environment(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING)
#endif
  {}

#ifdef USE_ONNX
    Ort::Env m_environment;
    std::unique_ptr<Ort::Session> m_classifier_session;
    std::unique_ptr<Ort::Session> m_initializer_session;
#endif
} ;

void
PTFlash::initONNX(std::string const& classifier_model_path,
                  std::string const& initializer_model_path,
                  eModelDNNType model,
                  int num_phase,
                  int num_compo)
{
  m_num_phase = num_phase ;
  m_num_compo = num_compo ;
  m_use_fp32 = true ;

  m_perf_mng.init("ONNX::Init") ;
  m_perf_mng.init("ONNX::Prepare") ;
  m_perf_mng.init("ONNX::Compute") ;
  m_perf_mng.init("ONNX::End") ;

#ifdef USE_ONNX
  m_onnx_internal = new ONNXInternal() ;
  m_use_onnx = true ;

  m_perf_mng.start("ONNX::Prepare") ;
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  // onnx session:
  //std::string classifier_onnx_file_name(classifier_model_path.localstr());
  switch(model)
  {
  case Classifier :
     m_classifier_model_path = classifier_model_path ;
     std::cout<<"ONNX INITIALIZING CLASSIFIER SESSION : "<<m_classifier_model_path<<std::endl ;
     m_onnx_internal->m_classifier_session = std::make_unique<Ort::Session>(m_onnx_internal->m_environment, m_classifier_model_path.c_str(), session_options);
     std::cout<<"ONNX Classifier Session OK"<<std::endl;
     break ;
  case Initializer :
     m_initializer_model_path = initializer_model_path ;
     std::cout<<"ONNX INITIALIZING INITIALIZER SESSION : "<<m_initializer_model_path<<std::endl ;
     m_onnx_internal->m_initializer_session = std::make_unique<Ort::Session>(m_onnx_internal->m_environment, m_initializer_model_path.c_str(), session_options);
     std::cout<<"ONNX Initialize Session OK"<<std::endl;
     break ;
  case FullClassInit :
     m_classifier_model_path = classifier_model_path ;
     std::cout<<"ONNX INITIALIZING CLASSIFIER SESSION : "<<m_classifier_model_path<<std::endl ;
     m_onnx_internal->m_classifier_session = std::make_unique<Ort::Session>(m_onnx_internal->m_environment, m_classifier_model_path.c_str(), session_options);
     std::cout<<"ONNX Classifier Session OK"<<std::endl;
     m_initializer_model_path = initializer_model_path ;
     std::cout<<"ONNX INITIALIZING INITIALIZER SESSION : "<<m_initializer_model_path<<std::endl ;
     m_onnx_internal->m_initializer_session = std::make_unique<Ort::Session>(m_onnx_internal->m_environment, m_initializer_model_path.c_str(), session_options);
     std::cout<<"ONNX Initialize Session OK"<<std::endl;
  }

  m_perf_mng.stop("ONNX::Prepare") ;
  // onnx session:
  //std::string initializer_onnx_file_name(initializer_model_path.localstr());
#endif
}

void PTFlash::_endComputeONNX() const
{
#ifdef USE_ONNX
  std::cout<<" COMPUTE ONNX"<<std::endl ;
  m_perf_mng.start("ONNX::Compute") ;

  // onnx tensors:
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  // input:
  std::vector<Ort::Value> input_tensors;
  Ort::TypeInfo input_type_info = m_onnx_internal->m_classifier_session->GetInputTypeInfo(0);
  auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> input_dims = input_tensor_info.GetShape();
  // batch size
  input_dims[0]=m_batch_size;
  input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(m_xf.data()),
                                                          m_xf.size(), input_dims.data(), input_dims.size()));

  // output:
  std::vector<Ort::Value> output_tensors;
  Ort::TypeInfo output_type_info = m_onnx_internal->m_classifier_session->GetOutputTypeInfo(0);
  auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> output_dims = output_tensor_info.GetShape();

  // batch size
  output_dims[0]=m_batch_size;
  output_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info,
                                                           m_yf.data(),
                                                           m_yf.size(),
                                                           output_dims.data(), output_dims.size()));

  // serving names:
  std::string input_name = m_onnx_internal->m_classifier_session->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get();
  std::vector<const char*>  input_names{input_name.c_str()};
  std::string output_name = m_onnx_internal->m_classifier_session->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get();
  std::vector<const char*> output_names{output_name.c_str()};

  std::cout<<" ONNX CLASSIFIER MODEL INFERENCE"<<std::endl ;
 // model inference
  m_onnx_internal->m_classifier_session->Run(Ort::RunOptions{nullptr},
                                             input_names.data(),
                                             input_tensors.data(), 1, output_names.data(),
                                             output_tensors.data(), 1);

  std::cout<<"onnx inference classifier ok"<<std::endl;

  m_current_index = 0 ;
  m_current_unstable_index = 0 ;
  m_unstable.resize(m_batch_size) ;
  m_theta_v.resize(m_batch_size) ;
  m_xi.resize(m_batch_size*m_num_compo) ;
  m_yi.resize(m_batch_size*m_num_compo) ;
  m_ki.resize(m_batch_size*m_num_compo) ;

  bool compute_ki = m_onnx_internal->m_initializer_session.get() != nullptr ;
  if(compute_ki)
  {
    m_x2f.resize(m_tensor_size*m_batch_size) ;
  }
  int offset = 0 ;
  int unstable_offset = 0 ;
  int nb_unstable = 0 ;
  for(std::size_t i=0;i<m_batch_size;++i)
  {
      double prob = 1./(1+std::exp(-m_yf[i])) ;
      m_unstable[i] = prob > 0.5 ;
      //std::cout<<"UNSTABLE["<<i<<"]"<<m_unstable[i]<<" "<<prob<<" "<<m_yf[i]<<std::endl ;

      if(compute_ki && m_unstable[i])
      {
          for(int j=0;j<m_tensor_size;++j)
            m_x2f[unstable_offset+j] = m_xf[offset+j] ;
          unstable_offset += m_tensor_size ;
          ++ nb_unstable ;
      }
      offset += m_tensor_size ;
  }

  std::cout<<" ONNX INITIALIZER MODEL INFERENCE"<<compute_ki<<" "<<nb_unstable<<std::endl ;
  if(compute_ki && nb_unstable>0)
  {
      m_y2f.resize(nb_unstable*m_num_compo) ;
      // input:
      std::vector<Ort::Value> input_tensors;
      Ort::TypeInfo input_type_info = m_onnx_internal->m_initializer_session->GetInputTypeInfo(0);
      auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> input_dims = input_tensor_info.GetShape();
      // batch size
      input_dims[0]=nb_unstable;
      input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(m_x2f.data()),
                                                              m_x2f.size(), input_dims.data(), input_dims.size()));

      // output:
      std::vector<Ort::Value> output_tensors;
      Ort::TypeInfo output_type_info = m_onnx_internal->m_initializer_session->GetOutputTypeInfo(0);
      auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> output_dims = output_tensor_info.GetShape();

      // batch size
      output_dims[0]=nb_unstable;
      output_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info,
                                                               m_y2f.data(),
                                                               m_y2f.size(),
                                                               output_dims.data(), output_dims.size()));

      // serving names:
      std::string input_name = m_onnx_internal->m_initializer_session->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get();
      std::vector<const char*>  input_names{input_name.c_str()};
      std::string output_name = m_onnx_internal->m_initializer_session->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get();
      std::vector<const char*> output_names{output_name.c_str()};

      std::cout<<" ONNX INITIALIZER MODEL INFERENCE"<<std::endl ;
     // model inference
      m_onnx_internal->m_initializer_session->Run(Ort::RunOptions{nullptr},
                                                 input_names.data(),
                                                 input_tensors.data(), 1, output_names.data(),
                                                 output_tensors.data(), 1);

      std::cout<<"onnx inference initializer ok"<<std::endl;

      for(std::size_t i=0;i<nb_unstable*m_num_compo;++i)
      {
        m_ki[i] = m_y2f[i] ;
      }
      /*
      std::cout<<"==================================================="<<std::endl ;
      std::cout<<"KI OUTPUT : "<<std::endl ;
      int offset = 0 ;
      for(int i = 0;i<nb_unstable;++i)
      {
         std::cout<<"KI["<<i<<"] : (";
         for(int j=0;j<m_num_compo;++j)
         {
           std::cout<<m_ki[offset+j]<<",";
         }
         offset += m_num_compo ;
         std::cout<<")"<<std::endl ;
      }*/
  }
  m_perf_mng.stop("ONNX::Compute") ;
#else
  std::cerr<<"ERROR : ONNX RUNTIME IS NOT AVAILABLE"<<std::endl ;
#endif

}

struct PTFlash::TensorRTInternal
{
  std::unique_ptr<TRTEngine> m_classifier_engine ;
  std::unique_ptr<TRTEngine> m_initializer_engine ;
} ;


void
PTFlash::initTensorRT(std::string const& classifier_model_path,
                      std::string const& initializer_model_path,
                      PTFlash::eModelDNNType model,
                      int num_phase,
                      int num_compo,
                      bool use_fp32)
{
  m_num_phase = num_phase ;
  m_num_compo = num_compo ;
  m_use_fp32 = use_fp32 ;

  m_classifier_model_path = classifier_model_path ;
  m_initializer_model_path = initializer_model_path ;

  m_perf_mng.init("TensorRT::Init") ;
  m_perf_mng.init("TensorRT::Prepare") ;
  m_perf_mng.init("TensorRT::Compute") ;
  m_perf_mng.init("TensorRT::End") ;
#ifdef USE_TENSORRT
  m_tensorrt_internal = new TensorRTInternal() ;
  m_model_type = model ;
  m_use_tensorrt = true ;
#endif

}


void PTFlash::_endComputeTensorRT() const
{
#ifdef USE_TENSORRT
    m_perf_mng.start("TensorRT::Compute") ;
    bool infer_status = m_tensorrt_internal->m_classifier_engine->infer(m_xf,m_yf,m_batch_size) ;
    if(!infer_status)
    {
        std::cerr<<"CLASSIFIER INFER FAILED"<<std::endl ;;
    }
    else
    {
          std::cerr<<"CLASSIFIER INFER OK"<<std::endl ;;
          bool compute_ki = m_tensorrt_internal->m_initializer_engine.get() != nullptr ;
          if(compute_ki)
          {
            if(m_use_fp32)
            {
              m_x2f.resize(0) ;
              m_x2f.reserve(m_tensor_size*m_batch_size) ;
            }
            else
            {
              m_x2.resize(0) ;
              m_x2.reserve(m_tensor_size*m_batch_size) ;
            }
          }
          int offset = 0 ;
          int unstable_offset = 0 ;
          int nb_unstable = 0 ;

          m_current_index = 0 ;
          m_current_unstable_index = 0 ;
          m_unstable.resize(m_batch_size) ;
          m_theta_v.resize(m_batch_size) ;
          m_xi.resize(m_batch_size*m_num_compo) ;
          m_yi.resize(m_batch_size*m_num_compo) ;
          m_ki.resize(m_batch_size*m_num_compo) ;
          for(std::size_t i=0;i<m_batch_size;++i)
          {
              double prob = 1./(1+std::exp(-m_yf[i])) ;
              m_unstable[i] = prob > 0.5 ;
              if(compute_ki && m_unstable[i])
              {
                  if(m_use_fp32)
                  {
                    for(int j=0;j<m_tensor_size;++j)
                      m_x2f.push_back(m_xf[offset+j]) ;
                  }
                  else
                  {
                    for(int j=0;j<m_tensor_size;++j)
                      m_x2.push_back(m_x[offset+j]) ;
                  }
                  unstable_offset += m_tensor_size ;
                  ++ nb_unstable ;
              }
              offset += m_tensor_size ;
          }
          if(m_use_fp32)
            assert(m_x2f.size()==unstable_offset) ;
          else
            assert(m_x2.size()==unstable_offset) ;

          if(compute_ki && nb_unstable>0)
          {
              m_ki.resize(nb_unstable*m_num_compo) ;
              bool infer_status = false ;
              if(m_use_fp32)
              {
                m_y2f.resize(nb_unstable*m_num_compo) ;
                infer_status = m_tensorrt_internal->m_initializer_engine->infer(m_x2f,m_y2f,nb_unstable) ;
                if(infer_status)
                {
                   for(std::size_t i=0;i<nb_unstable*m_num_compo;++i)
                   {
                     m_ki[i] = m_y2f[i] ;
                   }
                }
              }
              else
              {
                infer_status = m_tensorrt_internal->m_initializer_engine->infer(m_x2,m_ki,nb_unstable) ;
              }
              if(infer_status)
              {
                std::cout<<"INITIALIZER INFER OK"<<std::endl ;
                /*
                std::cout<<"==================================================="<<std::endl ;
                std::cout<<"KI OUTPUT : "<<std::endl ;
                int offset = 0 ;
                for(int i = 0;i<nb_unstable;++i)
                {
                   std::cout<<"KI["<<i<<"] : (";
                   for(int j=0;j<m_num_compo;++j)
                   {
                     std::cout<<m_ki[offset+j]<<",";
                   }
                   offset += m_num_compo ;
                   std::cout<<")"<<std::endl ;
                }*/
              }
              else
              {
                 std::cerr<<"INITIALIZER INFER FAILED"<<std::endl ;;
              }
          }
    }
    m_perf_mng.stop("TensorRT::Compute") ;
#endif
}

void PTFlash::startCompute(int batch_size) const
{
  m_batch_size = batch_size ;
  m_tensor_size = 2 + m_num_compo ;
  m_offset = 0 ;
  m_current_index = 0 ;
  m_current_unstable_index = 0 ;

  if(m_use_fp32)
  {
    m_xf.resize(m_tensor_size*m_batch_size) ;
    if(m_use_onnx || m_use_tensorrt)
      m_yf.resize(m_batch_size) ;
  }
  else
    m_x.resize(m_tensor_size*m_batch_size) ;

  if(m_use_tensorrt)
  {
     m_perf_mng.start("TensorRT::Prepare") ;
#ifdef USE_TENSORRT
     if(m_model_type == Classifier || m_model_type == FullClassInit)
     {
       std::cout<<"START TENSORRT CLASSIFIER BUILD"<<std::endl ;
       m_tensorrt_internal->m_classifier_engine.reset( new TRTEngine(m_classifier_model_path,m_batch_size,m_tensor_size,1));
       bool ok = m_tensorrt_internal->m_classifier_engine->build() ;
       if(!ok)
       {
          std::cerr<<"CLASSIFIER BUILD FAILED"<<std::endl ;;
       }
     }
     if(m_model_type == Initializer || m_model_type == FullClassInit)
     {
       std::cout<<"START TENSORRT INITIALIZER BUILD"<<std::endl ;
       m_tensorrt_internal->m_initializer_engine.reset( new TRTEngine(m_initializer_model_path,m_batch_size,m_tensor_size,1));
       bool ok = m_tensorrt_internal->m_initializer_engine->build() ;
       if(!ok)
       {
          std::cerr<<"INITIALIZER BUILD FAILED"<<std::endl ;;
       }
     }
#endif
     m_perf_mng.stop("TensorRT::Prepare") ;
  }
}


void PTFlash::asynchCompute(const double P,
                           const double T,
                           std::vector<double> const& Zk) const
{

  if(m_use_fp32)
  {
    m_xf[m_offset + 0] = P ;
    m_xf[m_offset + 1] = T ;
    for(int ic=0;ic<m_num_compo;++ic)
    {
      m_xf[m_offset + 2 + ic] = Zk[ic] ;
    }
  }
  else
  {
    m_x[m_offset + 0] = P ;
    m_x[m_offset + 1] = T ;
    for(int ic=0;ic<m_num_compo;++ic)
    {
      m_x[m_offset + 2 + ic] = Zk[ic] ;
    }
  }
  m_offset += m_tensor_size ;
  /*
  std::cout<<"[P,T,Zk]=["<<P<<","<<T<<",";
  for(int ic=0;ic<m_num_compo;++ic)
  {
    std::cout<<Zk[ic]<<",";
  }
  std::cout<<"]"<<std::endl ;
  */
}


void PTFlash::endCompute() const
{

  if(m_use_carnot)
  {
    _endComputeCarnot() ;
  }

  if(m_use_cawf_inference)
  {
    _endComputeCAWF() ;
  }

  if(m_use_torch)
  {
     _endComputeTorch() ;
  }

  if(m_use_onnx)
  {
     _endComputeONNX() ;
  }

  if(m_use_tensorrt)
  {
     _endComputeTensorRT() ;
  }

  m_perf_mng.printInfo();

  m_current_index = 0 ;
  m_current_unstable_index = 0 ;
}

bool PTFlash::getNextResult(std::vector<double>& Thetap,
                            std::vector<double>& Xkp) const
{
  bool unstable = m_unstable[m_current_index++] ;
  if(unstable)
  {
      Thetap[0] = m_theta_v[m_current_unstable_index] ;
      Thetap[1] = 1 - Thetap[0];

      for (int icompo = 0; icompo < m_num_compo; icompo++)
      {
          Xkp[icompo]             = m_xi[m_current_unstable_index*m_num_compo+icompo];
          Xkp[m_num_compo+icompo] = m_yi[m_current_unstable_index*m_num_compo+icompo] ;
      }
      ++m_current_unstable_index ;
  }
  else
  {
    Thetap[0] = 1. ;
    Thetap[1] = 0. ;
    for (int icompo = 0; icompo < m_num_compo; icompo++)
    {
        Xkp[icompo]             = 0.;
        Xkp[m_num_compo+icompo] = 0. ;
    }
  }
  return unstable ;
}

