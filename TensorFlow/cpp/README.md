# Tutorial TensorFlow CPP inference

## BUILD

```bash
source eb.env
module load GCC/9.3.0
module load CUDA/11.0.2
module load CMake/3.16.4
export TENSORFLOW_ROOT="path_to_tensorflow_2"

mkdir build
cd build
cmake ..
cmake test
```