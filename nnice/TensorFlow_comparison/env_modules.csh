ml GCC/8.3.0 OpenMPI/3.1.4 TensorFlow/2.3.1-Python-3.7.4

setenv CXX g++
setenv TF_INCLUDE /soft/irsrvsoft1/expl/eb/centos_7/software/MPI/GCC/8.3.0/OpenMPI/3.1.4/TensorFlow/2.3.1-Python-3.7.4/API_C++/include
setenv TF_LIB /soft/irsrvsoft1/expl/eb/centos_7/software/MPI/GCC/8.3.0/OpenMPI/3.1.4/TensorFlow/2.3.1-Python-3.7.4/API_C++/lib
setenv TF_LFLAG tensorflow_cc

setenv OMP_PROC_BIND MASTER
setenv TF_CPP_MIN_LOG_LEVEL 3
setenv I_MPI_PIN_PROCESSOR_LIST 0-1