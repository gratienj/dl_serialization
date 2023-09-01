/*
 * PTFlash.h
 *
 *  Created on: 3 août 2023
 *      Author: gratienj
 */

#pragma once

#include <vector>

#ifdef USE_CAWFINFERENCE
#include <cawf_inference/cawf_api.h>
#endif

class PTFlash
{
public:

    struct CarnotInternal ;

    PTFlash(int output_level=0)
    : m_output_level(output_level)
    {}

    void initCarnot(std::string const& carnot_model_path,
                    int num_phase,
                    int num_compo,
                    std::vector<int> const& component_uids) ;

    void initCAWF(std::string const& cawf_model_path,
                    int num_phase,
                    int num_compo) ;

    void end() ;

    void startCompute(int batch_size) const;
    void asynchCompute(const double P,
                       const double T,
                       std::vector<double> const& Zk) const ;
    void endCompute() const ;
    bool getNextResult(std::vector<double>& Thetap,
                       std::vector<double>& Xkp) const ;
private:
    int m_output_level = 0 ;
    int m_num_phase    = 0 ;
    int m_num_compo    = 0 ;

    bool            m_use_gpu             = false ;
    bool            m_use_cawf_inference  = false ;
    bool            m_use_carnot          = false ;
#ifdef USE_CAWFINFERENCE
    mutable cawf_inference::CAWFInferenceMng m_cawf_inference_mng ;
#endif
    CarnotInternal* m_carnot_internal = nullptr ;


    mutable int64_t               m_batch_size             = 0 ;
    mutable int64_t               m_tensor_size            = 0 ;
    mutable int                   m_offset                 = 0 ;
    mutable int                   m_current_index          = 0 ;
    mutable int                   m_current_unstable_index = 0 ;

    std::size_t const             m_unstable_id = 0 ;
    std::size_t const             m_theta_v_id  = 0 ;
    std::size_t const             m_xi_id       = 1 ;
    std::size_t const             m_yi_id       = 2 ;
    std::size_t const             m_ki_id       = 3 ;

    mutable std::vector<double>   m_x ;
    mutable std::vector<float>    m_xf ;
    mutable std::vector<bool>     m_unstable ;
    mutable std::vector<double>   m_theta_v ;
    mutable std::vector<double>   m_xi ;
    mutable std::vector<double>   m_yi ;
    mutable std::vector<double>   m_ki ;

    mutable PerfCounterMng<std::string> m_perf_mng ;


};
