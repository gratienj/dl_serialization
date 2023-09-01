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


void PTFlash::startCompute(int batch_size) const
{
  m_batch_size = batch_size ;
  m_tensor_size = 2 + m_num_compo ;
   m_x.resize(m_tensor_size*m_batch_size) ;
  m_offset = 0 ;
  m_current_index = 0 ;
  m_current_unstable_index = 0 ;
}


void PTFlash::asynchCompute(const double P,
                           const double T,
                           std::vector<double> const& Zk) const
{

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
#ifdef USE_CARNOT
  if(m_use_carnot)
    {
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

          assert(all_fluids.size()==2) ;
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
            assert(all_fluids.size()==1) ;
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
    }
#endif

#ifdef USE_CAWFINFERENCE
  if(m_use_cawf_inference)
    {

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
    }
#endif
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

