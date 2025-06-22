#!/bin/bash

set -e  # stop on error

echo "Step 1: Install required system packages"
sudo apt update
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
  libnss3-dev libssl-dev libreadline-dev libffi-dev wget libbz2-dev \
  libsqlite3-dev liblzma-dev uuid-dev tk-dev

echo "Step 2: Download Python 3.10.9 source"
cd /tmp
wget https://www.python.org/ftp/python/3.10.9/Python-3.10.9.tgz
tar -xf Python-3.10.9.tgz
cd Python-3.10.9

echo "Step 3: Build and install Python 3.10.9"
./configure --enable-optimizations
make -j4
sudo make altinstall

echo "Python 3.10.9 installed."
