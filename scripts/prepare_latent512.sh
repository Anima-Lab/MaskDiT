# Encode ImageNet 512x512 into latent space

python3 extract_latent.py --resolution 512 --ckpt assets/vae/autoencoder_kl.pth --batch_size 64 --outdir ../data/imagenet512-latent

# Convert lmdb to webdataset
python3 lmdb2wds.py --maxcount 10010 --datadir ../data/imagenet512-latent --outdir ../data/imagenet512-latent-wds --resolution 64 --num_channels 8
