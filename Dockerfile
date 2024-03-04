FROM nvcr.io/nvidia/pytorch:23.03-py3
RUN pip install einops lmdb omegaconf wandb tqdm pyyaml accelerate
RUN pip install timm webdataset
RUN pip install diffusers["torch"] transformers