#!/bin/bash

# Run and execute commands for container 1
docker run -it -d -e "CUDA_VISIBLE_DEVICES=0" -e "CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps" -e "CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log" --name py1 --gpus '"device=0"' -v /tmp/nvidia-mps:/tmp/nvidia-mps --ipc=host pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
docker cp ~/resnet50_cifar10.py py1:/workspace

# Run and execute commands for container 2
docker run -it -d -e "CUDA_VISIBLE_DEVICES=0" -e "CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps" -e "CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log" --name py2 --gpus '"device=1"' -v /tmp/nvidia-mps:/tmp/nvidia-mps --ipc=host pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
docker cp ~/resnet50_cifar10.py py2:/workspace

# Run and execute commands for container 5
docker run -it -d -e "CUDA_VISIBLE_DEVICES=0,1" -e "CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps" -e "CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log" --name py3 --gpus all -v /tmp/nvidia-mps:/tmp/nvidia-mps --ipc=host pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
docker cp ~/resnet50_cifar10.py py3:/workspace
