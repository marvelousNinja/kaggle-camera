#!/bin/bash
set -e
set -v

# Install system deps for Python
sudo apt-get install -y unzip make build-essential \
             libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
             libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
             xz-utils tk-dev

curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash

# Install Python and project deps
pyenv install 3.6.3
pyenv virtualenv 3.6.3 kaggle-camera-3.6.3
pip3 install -r -requirements.txt
