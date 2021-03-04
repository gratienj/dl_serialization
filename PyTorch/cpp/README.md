# Tutorial PyTorch CPP inference

## BUILD

```bash
source eb.env
module load GCC/7.3.0-2.30
module load CUDA/10.0.130
module load OpenCV/4.0.1
module load CMake/3.16.2

export Torch_ROOT="path_to_libtorch"

mkdir build
cd build
cmake ..
cmake test
```