# Code for paper "Fast training of diffusion model with masked transformers"

## Requirements
- 8 A100 GPUs are needed for training. 
- At least one high-end GPU for sampling. 
- `Dockerfile` is provided for exact software environment. 

## Prepare dataset
We use the pre-trained VAE to first encode the ImageNet dataset into latent space. You can download the pre-trained VAE by using `download_assets.py`. 
```
python3 download_assets.py --name vae --dest assets
```
## Train
To train from scratch, run
```bash
python3 train_latent.py --config configs/train/maskdit-latent-base.yaml --num_process_per_node 8
```

We also provide code for training MaskDiT without pre-encoded dataset in `train.py`. This is only for reference. We did not test it. 


## Generate samples
To generate samples from provided checkpoints, for example, run
```bash
python3 generate.py --config configs/test/maskdit-latent-base.yaml --ckpt_path results/2075000.pt --class_idx 388 --cfg_scale 2.5
```
Checkpoints of MaskDiT can be downloaded by running `download_assets.py`. For example, 
```bash
python3 download_assets.py --name maskdit-finetune0 --dest results
```

## Evaluation
First, download the reference from [ADM repo](https://github.com/openai/guided-diffusion/tree/main/evaluations) directly. You can also use `download_assets.py` by running 
```bash
python3 download_assets.py --name imagenet256 --dest [destination directory]
```
Then we can use either the evaluator from [ADM repo](https://github.com/openai/guided-diffusion/tree/main/evaluations) or `fid.py` to evaluate the generated samples.