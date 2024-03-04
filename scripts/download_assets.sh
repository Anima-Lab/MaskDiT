# download pretrained VAE
python3 download_assets.py --name vae --dest assets/stable_diffusion

# download ImageNet256 training set
python3 download_assets.py --name imagenet256-latent-lmdb --dest ../data/imagenet256

# download ImageNet512 training set
python3 download_assets.py --name imagenet512-latent-wds --dest ../data/imagenet512-wds