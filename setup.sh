#!/bin/bash
set -e
set -v

# Add NVIDIA repo
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
rm ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

# Install CUDA and system deps for Python
sudo apt-get update && sudo apt-get install -y cuda-8.0 unzip make build-essential \
             libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
             libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
             xz-utils tk-dev libturbojpeg

# Instal CuDNN
curl -O http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/cudnn-8.0-linux-x64-v6.0.tgz
tar -xvf ./cudnn-8.0-linux-x64-v6.0.tgz -C ./
sudo cp -P ./cuda/lib64/* /usr/local/cuda/lib64
sudo cp ./cuda/include/* /usr/local/cuda/include
rm -rf ./cuda

# Set persistence mode
nvidia-smi -pm 1

# Install pyenv
curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash

# Setup env for pyenv and CUDA
export PYENV_ROOT="${HOME}/.pyenv"
echo "export PATH=\"${PYENV_ROOT}/bin:\$PATH\"" >> ~/.profile
echo "eval \"\$(pyenv init -)\"" >> ~/.profile
echo "eval \"\$(pyenv virtualenv-init -)\"" >> ~/.profile
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.profile
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.profile
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64' >> ~/.profile
source ~/.profile

# Install Python and project deps
pyenv install 3.6.3
pyenv virtualenv 3.6.3 kaggle-camera-3.6.3
pip3 install -r requirements.txt

# Replace TF with GPU version
pip3 uninstall -y tensorflow
pip3 --no-cache-dir install https://github.com/mind/wheels/releases/download/tf1.4.1-gpu/tensorflow-1.4.1-cp36-cp36m-linux_x86_64.whl
