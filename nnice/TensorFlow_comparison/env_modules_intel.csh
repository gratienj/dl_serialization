ml iimpi/2020a imkl/2020.1.217

setenv CXX mpicxx
setenv LD_LIBRARY_PATH /home/irsrvshare2/R10/VKM-CONVERGE/GIT/LIB/V3.0/centos7_foss2019a/TENSORFLOW-GCC/lib:$LD_LIBRARY_PATH

setenv TF_INCLUDE /home/irsrvshare2/R10/VKM-CONVERGE/GIT/LIB/V3.0/centos7_foss2019a/TENSORFLOW-GCC/include
setenv TF_LIB /home/irsrvshare2/R10/VKM-CONVERGE/GIT/LIB/V3.0/centos7_foss2019a/TENSORFLOW-GCC/lib
setenv TF_LFLAG tensorflow

setenv OMP_PROC_BIND MASTER
setenv TF_CPP_MIN_LOG_LEVEL 3
setenv I_MPI_PIN_PROCESSOR_LIST 0-1