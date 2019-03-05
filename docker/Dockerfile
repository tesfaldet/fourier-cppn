FROM tensorflow/tensorflow:1.13.1-gpu-py3

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Patch path
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64/stubs

ARG PIP_EXTRA_INDEX_URL
RUN pip install --upgrade pip
RUN pip install opencv-python lucid \
  eai-shuriken

WORKDIR /src
COPY . .

CMD ["python3", "train.py"]
