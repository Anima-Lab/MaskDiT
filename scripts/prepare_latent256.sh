# Encode ImageNet 256x256 into latent space

python3 extract_latent.py --resolution 256 --ckpt assets/vae/autoencoder_kl.pth --batch_size 64 --outdir ../data/imagenet256-latent
