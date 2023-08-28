#!/bin/bash

# Check if the required input variable is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <cuda_visible_devices> <default_active_thread_percentage>"
  exit 1
fi

# Set the input variables
CUDA_VISIBLE_DEVICES="$1"
DEFAULT_ACTIVE_THREAD_PERCENTAGE="$2"

# Set the CUDA MPS directories
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

# Set exclusive process mode for the GPU
sudo nvidia-smi -i $1 -c EXCLUSIVE_PROCESS

# Start the CUDA MPS control daemon
nvidia-cuda-mps-control -d

# Set the default active thread percentage
echo "set_default_active_thread_percentage $DEFAULT_ACTIVE_THREAD_PERCENTAGE" | nvidia-cuda-mps-control

echo "Script completed successfully."
