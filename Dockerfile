FROM nvcr.io/nvidia/pytorch:23.03-py3
RUN pip install lmdb omegaconf wandb tqdm pyyaml accelerate
RUN pip install git+https://github.com/huggingface/pytorch-image-models.git
RUN pip install git+https://github.com/huggingface/diffusers