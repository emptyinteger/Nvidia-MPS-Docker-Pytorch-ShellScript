# Nvidia-MPS-Docker-Pytorch-ShellScript

## pre-requieste and use

1. nvidia-driver installation
2. docker installation
3. nvidia-container-toolkit installation

upper step https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html 
  
5. launch mpsdaemon.sh
6. launch MPSdocker.sh
7. docker exec -it container-name bash
8. run pytorch learning

## docker installation

```curl https://get.docker.com | sh   && sudo systemctl --now enable docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)       && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg       && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list |             sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |             sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo usermod -aG docker deepl
sudo reboot now
```

## 사용법

아래는 `mpsdaemon.sh` 스크립트의 사용 예시입니다.

```bash
# 단일 GPU 사용 예시
sudo sh mpsdaemon.sh 0 10

# 다중 GPU 사용 예시
sudo sh mpsdaemon.sh 0,1 10
sudo sh mpsdaemon.sh 0,1,2,3,... 10
sudo sh mpsdaemon.sh 0,1,2,3,... 33
```

args 1 means the NVIDIA_DEVICE_ID , <br>
args 2 means the default thread share for one process


## 컨테이너 실행 및 명령어 실행

If you want to use ```resnet.py``` then replace ```resnet50_cifar10.py``` to ```resnet.py``` under below script

```bash
### 컨테이너 1 which select device 0 to use
docker run -it -d -e "CUDA_VISIBLE_DEVICES=0" -e "CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps" -e "CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log" --name py1 --gpus '"device=0"' -v /tmp/nvidia-mps:/tmp/nvidia-mps --ipc=host pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
docker cp ~/resnet50_cifar10.py py1:/workspace

### 컨테이너 2 which select device 1 to use
docker run -it -d -e "CUDA_VISIBLE_DEVICES=0" -e "CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps" -e "CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log" --name py2 --gpus '"device=1"' -v /tmp/nvidia-mps:/tmp/nvidia-mps --ipc=host pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
docker cp ~/resnet50_cifar10.py py2:/workspace

### 컨테이너 3 which select ALL to use
docker run -it -d -e "CUDA_VISIBLE_DEVICES=0,1" -e "CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps" -e "CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log" --name py3 --gpus all -v /tmp/nvidia-mps:/tmp/nvidia-mps --ipc=host pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
docker cp ~/resnet50_cifar10.py py3:/workspace
```
### Usage

아래는 `MPSdocker.sh` 스크립트의 사용 예시입니다.

```bash
sh MPSdocker.sh
```

## how to execute resnet50_cifar10.py

Under instruction to train `resnet50_cifar10.py` in py1(your container name) 

```bash
docker exec -it py1(your container name) bash
pip install matplotlib
torchrun --standalone resnet50_cifar10.py
```

## how to execute resnet.py

Under instruction to train `resnet50_cifar10.py` in py1(your container name) 

```bash
docker exec -it py1(your container name) bash
python3 resnet.py
```

## if you have trouble in training

try under command and restart `mpsdaemon.sh`

```bash
sudo pkill -9 nvidia-cuda-mps
```

## Result

<img width="446" alt="image" src="https://github.com/emptyinteger/Nvidia-MPS-Docker-Pytorch-ShellScript/assets/92441821/2fbb5ffe-e59a-4b94-a286-d7a3f59374a5">
