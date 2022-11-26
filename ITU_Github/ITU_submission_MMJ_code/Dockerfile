# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
## Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
#FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu18.04
#nvcr.io/nvidia/pytorch:22.01-py3
FROM nvidia/cuda:11.2.2-runtime-ubuntu18.04

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx python3 python3-pip

# Install python dependencies
COPY requirements.txt .

#RUN python -m pip install --upgrade pip
#RUN pip3 uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip3 install --no-cache -r requirements.txt albumentations coremltools onnx gsutil notebook numpy Pillow wandb>=0.12.2
RUN pip3 install torch==1.5.1+cu112 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip install --no-cache torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
RUN pip3 install torch==1.7.1+cu112 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --upgrade nni
RUN pip3 install --upgrade h5py
#torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# RUN pip install --no-cache -U torch torchvision

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf /root/.config/Ultralytics/
RUN pip install tensorflow
RUN pip install netron
# Set environment variables
# ENV HOME=/usr/src/app


# Usage Examples -------------------------------------------------------------------------------------------------------

# Build and Push
# t=ultralytics/yolov5:latest && sudo docker build -t $t . && sudo docker push $t

# Pull and Run
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all $t

# Pull and Run with local directory access
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/datasets:/usr/src/datasets $t

# Kill all
# sudo docker kill $(sudo docker ps -q)

# Kill all image-based
# sudo docker kill $(sudo docker ps -qa --filter ancestor=ultralytics/yolov5:latest)

# Bash into running container
# sudo docker exec -it 5a9b5863d93d bash

# Bash into stopped container
# id=$(sudo docker ps -qa) && sudo docker start $id && sudo docker exec -it $id bash

# Clean up
# docker system prune -a --volumes

# Update Ubuntu drivers
# https://www.maketecheasier.com/install-nvidia-drivers-ubuntu/

# DDP test
# python -m torch.distributed.run --nproc_per_node 2 --master_port 1 train.py --epochs 3

# GCP VM from Image
# docker.io/ultralytics/yolov5:latest
