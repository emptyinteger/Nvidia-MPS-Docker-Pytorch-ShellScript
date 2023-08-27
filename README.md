# Nvidia-MPS-Docker-Pytorch-ShellScript

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

args 1 means the NVIDIA_DEVICE_ID 
args 2 means the default thread share for one process
