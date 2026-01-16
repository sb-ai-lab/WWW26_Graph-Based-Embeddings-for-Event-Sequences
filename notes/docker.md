# Docker

Each time there's a push to the main a docker image is created and pushed [here](This/Url/Is/Removed/Until/The/Review/Process/Finishes)

The docker image contains the dependencies to run the experiments. However, docker was not used for the most of the experiments due to the restrictions on our computing cluster


## Dependencies to be installed on host
Before building the dockerfile you should
1. Install cuda toolkit 12.1.0 or later on your host machine ([follow official instructions](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=deb_local))
2. Install NVIDIA Container Toolkit on your host machine (instructions are below)

NVIDIA Container Toolkit installation instrucitons: 

```python
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install the NVIDIA Docker package
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart the Docker daemon to apply the changes
sudo systemctl restart docker

```

## Build
```sh
docker build -t your_tag .
```
## Run
```sh
docker run --gpus all --ipc=host -it --rm -p 6006:6006 -p 8082:8082 -p 4041:4041 your_tag 
```


# Known problems

Хоть `bin/make_datasets_spark_file.sh` корректно отрабатывает, процесс невозможно отследить через `localhost:4041`. При запуске вне контейнера по этому порту доступен UI, демонстррующий статус задачи.
