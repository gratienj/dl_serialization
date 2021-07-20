# Tutorial TensorFlow CPP inference

## DEPENDANCIES

- TensorFlow
- cppflow (to install in the TensorFlow include directory)
- cnpy

## BUILD

```bash
source eb.env
module load GCC/9.3.0
module load CUDA/11.0.2
module load CMake/3.16.4
module load OpenMPI/4.0.3
module load Boost/1.72.0
export TENSORFLOW_ROOT="path_to_tensorflow_2"

mkdir build
cd build
cmake ..
cmake test
```