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
    struct TorchInternal ;
    struct ONNXInternal ;
    struct TensorRTInternal ;

    typedef enum {
        Classifier,
        Initializer,
        FullClassInit,
        FullFlash,
        eUndefinedModel
    } eModelDNNType ;

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

    void initTorch(std::string const& ptflash_model_path,
                   int num_phase,
                   int num_compo,
                   bool use_gpu=false) ;

    void initTorch(std::string const& classifier_model_path,
                   std::string const& initializer_model_path,
                   eModelDNNType model,
                   int num_phase,
                   int num_compo,
                   bool use_gpu=false,
                   bool use_fp32=false) ;

    void initONNX(std::string const& classifier_model_path,
                  std::string const& initializer_model_path,
                  eModelDNNType model,
                  int num_phase,
                  int num_compo) ;

    void initTensorRT(std::string const& classifier_model_path,
                      std::string const& initializer_model_path,
                      eModelDNNType model,
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
    void _endComputeCarnot() const ;
    void _endComputeCAWF() const ;
    void _endComputeTorch() const ;
    void _endComputeONNX() const ;
    void _endComputeTensorRT() const ;

    int m_output_level = 0 ;
    int m_num_phase    = 0 ;
    int m_num_compo    = 0 ;

    bool            m_use_gpu             = false ;
    bool            m_use_fp32            = false ;
    bool            m_use_cawf_inference  = false ;
    bool            m_use_carnot          = false ;
    bool            m_use_torch           = false ;
    bool            m_use_onnx            = false ;
    bool            m_use_tensorrt        = false ;


#ifdef USE_CAWFINFERENCE
    mutable cawf_inference::CAWFInferenceMng m_cawf_inference_mng ;
#endif
    CarnotInternal*   m_carnot_internal   = nullptr ;
    TorchInternal*    m_torch_internal    = nullptr ;
    ONNXInternal*     m_onnx_internal     = nullptr ;
    TensorRTInternal* m_tensorrt_internal = nullptr ;
    std::string       m_classifier_model_path ;
    std::string       m_initializer_model_path ;
    std::string       m_flash_model_path ;
    eModelDNNType     m_model_type = eUndefinedModel ;


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
    mutable std::vector<double>   m_x2 ;
    mutable std::vector<float>    m_xf ;
    mutable std::vector<float>    m_x2f ;
    mutable std::vector<float>    m_yf ;
    mutable std::vector<float>    m_y2f ;

    mutable std::vector<bool>     m_unstable ;
    mutable std::vector<double>   m_theta_v ;
    mutable std::vector<double>   m_xi ;
    mutable std::vector<double>   m_yi ;
    mutable std::vector<double>   m_ki ;

    mutable PerfCounterMng<std::string> m_perf_mng ;


};
