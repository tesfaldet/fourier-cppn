FROM nvcr.io/nvidia/tensorflow:19.01-py3

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*


ARG PIP_EXTRA_INDEX_URL
RUN pip install --upgrade pip
RUN pip install opencv-python lucid \
  eai-shuriken

WORKDIR /src
COPY . .

CMD ["python3", "train.py"]
