ml GCC/7.3.0-2.30  OpenMPI/3.1.1

setenv LD_LIBRARY_PATH /home/irsrvshare2/R10/VKM-CONVERGE/users/mehlc/tensor_flow_installation/lib:$LD_LIBRARY_PATH

setenv CXX g++
setenv TF_INCLUDE /home/irsrvshare2/R10/VKM-CONVERGE/users/mehlc/tensor_flow_installation/include
setenv TF_LIB /home/irsrvshare2/R10/VKM-CONVERGE/users/mehlc/tensor_flow_installation/lib
setenv TF_LFLAG tensorflow

setenv OMP_PROC_BIND MASTER
setenv TF_CPP_MIN_LOG_LEVEL 3
setenv I_MPI_PIN_PROCESSOR_LIST 0-1