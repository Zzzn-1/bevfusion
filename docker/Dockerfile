FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

RUN apt-get update && apt-get install wget -yq
RUN apt-get install build-essential g++ gcc -y
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y


# lzn添加
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
     apt-get update && apt-get upgrade -y --fix-missing
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y --fix-missing
RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y --fix-missing



ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

# 核心修改1：下载Python 3.8专属的Miniconda版本（避免版本冲突）
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR} && \
     # 删除安装脚本，节省空间
     rm -rf ~/miniconda.sh && \
     # 添加清华源（仅保留Python 3.8兼容的源）
     conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
     conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/ && \
     conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ && \
     conda config --set show_channel_urls yes && \
     # 核心修改3：确认Python 3.8（base环境已自带，无需重新安装）
     conda install python=3.8 -y --force-reinstall 


# 安装pytorch（适配Python 3.8+CUDA 11.3）
RUN conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -y 

# pip安装Pillow（换清华源）
RUN pip install Pillow==8.4.0 -i https://pypi.tuna.tsinghua.edu.cn/simple




# # Install miniconda
# ENV CONDA_DIR /opt/conda

# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#      /bin/bash ~/miniconda.sh -b -p /opt/conda
# # Put conda in path so we can use conda activate
# ENV PATH=${CONDA_DIR}/bin:${PATH}  
# RUN conda tos accept && \
#      # 添加清华conda源（加速国内下载，保留）
#      conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
#      conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/ && \
#      conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ && \
#      conda config --set show_channel_urls yes && \
#      # 安装Python 3.8：用-y替代--yes（通用参数）
#      conda install python=3.8 -y


# RUN conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
# RUN pip install Pillow==8.4.0
RUN pip install tqdm
RUN pip install torchpack
RUN pip install mmcv==1.4.0 mmcv-full==1.4.0 mmdet==2.20.0
RUN pip install nuscenes-devkit
RUN pip install mpi4py==3.0.3
RUN pip install numba==0.48.0
