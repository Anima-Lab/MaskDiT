FROM nvcr.io/nvidia/pytorch:23.03-py3
RUN pip install lmdb omegaconf wandb tqdm pyyaml accelerate
RUN pip install timm 
RUN pip install diffusers["torch"] transformers