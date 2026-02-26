import torch
from torch.utils.cpp_extension import load
import os

# 替换为你的GPU架构（比如compute_75）
GPU_ARCH = "compute_86"
SM_ARCH = "sm_86"

feature_decorator_ext = load(
    name="feature_decorator_ext",  # 扩展名
    sources=[
        os.path.join("src", "feature_decorator.cpp"),
        os.path.join("src", "feature_decorator_cuda.cu")
    ],  # 源码路径
    extra_cflags=["-O3", "-std=c++14", "-D_GLIBCXX_USE_CXX11_ABI=0"],  # C++编译参数
    extra_cuda_cflags=[
        "-O3", "-std=c++14",
        f"-gencode=arch={GPU_ARCH},code={GPU_ARCH}",  # 仅保留你的GPU架构
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
    ],  # CUDA编译参数
    verbose=True,  # 输出详细编译日志，方便排查
    build_directory="./build"  # 编译缓存目录
)

