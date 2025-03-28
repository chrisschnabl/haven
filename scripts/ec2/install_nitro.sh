#!/bin/bash

sudo dnf install aws-nitro-enclaves-cli -y
sudo dnf install aws-nitro-enclaves-cli-devel -y
sudo usermod -aG ne ec2-user
sudo usermod -aG docker ec2-user
# nitro-cli --version
sudo systemctl enable --now nitro-enclaves-allocator.service
sudo systemctl enable --now docker

sudo usermod -aG ne ec2-user
sudo usermod -aG docker ec2-user

sudo yum install -y llvm llvm-devel clang cmake
sudo yum groupinstall "Development Tools" -y
