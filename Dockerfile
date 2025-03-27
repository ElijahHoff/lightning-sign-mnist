FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    pytorch-lightning==2.0.9 \
    torchmetrics==0.11.4 \
    regex==2023.6.3

WORKDIR /workspace
COPY . /workspace

CMD ["python", "M1_PyTorchLightining_practice.py"]
