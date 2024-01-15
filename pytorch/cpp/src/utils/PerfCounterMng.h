/*
 * PerfCountMng.h
 *
 *  Created on: Jan 26, 2012
 *      Author: gratienj
 */

#pragma once

//#include <boost/timer.hpp>
#include <map>
#include <iostream>
#include <iomanip>
#include "utils/rdtsc.h"
#include <chrono>
#include <ctime>

template<typename PhaseT, bool use_tick=false>
class PerfCounterMng
{
public:
  typedef unsigned long long int         ValueType ;
  typedef PhaseT                         PhaseType ;
  typedef std::pair<ValueType,ValueType> CountType ;
  typedef std::map<PhaseType,CountType>  CountListType ;
  typedef PerfCounterMng<PhaseType>      ThisType ;

  class Sentry
  {
  public :
    Sentry(ThisType& parent, PhaseType phase)
    : m_parent(parent)
    , m_phase(phase)
    {
      m_parent.start(m_phase) ;
    }

    ~Sentry()
    {
      release() ;
    }

    void release()
    {
      m_parent.stop(m_phase) ;
    }

  private:
    ThisType& m_parent ;
    PhaseType m_phase ;
  };


  PerfCounterMng()
  : m_last_value(0)
  {
    m_cpu_frec = getCpuFreq() ;
    if(m_cpu_frec==0) m_cpu_frec = 1. ;
    m_t0 = std::chrono::system_clock::now() ;
    if(use_tick)
      m_factor = 1.e-6/m_cpu_frec ;
    else
      m_factor = 1.e-6 ;
  }

  virtual ~PerfCounterMng(){}

  void init(PhaseType const& phase) {
    CountType& count = m_counts[phase] ;
    count.first = 0 ;
    count.second = 0 ;
  }

  void getCount(ValueType& value) {
    if(use_tick)
      rdtsc(&value) ;
    else
    {
      std::chrono::time_point<std::chrono::system_clock> t = std::chrono::system_clock::now() ;
      value = std::chrono::duration_cast<std::chrono::microseconds>(t-m_t0).count() ;
    }
  }

  void start(PhaseType const& phase)
  {
    getCount(m_counts[phase].second) ;
  }

  void stop(PhaseType const& phase)
  {
    CountType& count = m_counts[phase] ;
    getCount(m_last_value) ;
    m_last_value = m_last_value - count.second ;
    count.first += m_last_value ;
  }

  ValueType getLastValue() {
    return m_last_value ;
  }

  ValueType getValue(PhaseType const& phase) {
    return m_counts[phase].first ;
  }

  double getValueInSeconds(PhaseType const& phase) {
    return m_counts[phase].first*m_factor;
  }

  void printInfo() const {
    std::cout<<"PERF INFO : "<<std::endl ;
    std::cout<<std::setw(20)<<"COUNT"<<" : "<<"VALUE"<<std::endl ;
    for(typename CountListType::const_iterator iter = m_counts.begin();iter!=m_counts.end();++iter)
    {
      std::cout<<std::setw(20)<<(*iter).first<< " : "<<(*iter).second.first*m_factor<<'\n' ;
    }
  }

  void printInfo(std::ostream& stream) const {
    stream<<"PERF INFO : "<<std::endl ;
    stream<<std::setw(10)<<"COUNT"<<" : "<<"VALUE"<<std::endl ;
    for(typename CountListType::const_iterator iter = m_counts.begin();iter!=m_counts.end();++iter)
    {
      stream<<std::setw(10)<<(*iter).first<< " : "<<(*iter).second.first*m_factor<<'\n' ;
    }
  }

  int getCpuFreq()
  {
    return 2700 ;
    /* return cpu frequency in MHZ as read in /proc/cpuinfo */
    float ffreq = 0;
    int r = 0;
    char *rr = NULL;
#ifndef WIN32
    FILE *fdes = fopen("/proc/cpuinfo","r");
    char buff[256];
    size_t bufflength = 256;
    do{
      rr = fgets(buff,bufflength,fdes);
      r = sscanf(buff,"cpu MHz         : %f\n",&ffreq);
      if(r==1){
        break;
      }
    }while(rr != NULL);

    fclose(fdes);
#endif

    int ifreq = ffreq;
    return ifreq;
  }

private :
  ValueType     m_last_value ;
  CountListType m_counts ;
  double        m_cpu_frec = 1.;
  double        m_factor     = 1.0e-6;
  std::chrono::time_point<std::chrono::system_clock> m_t0 ;
};

