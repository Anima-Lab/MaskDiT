accelerate launch \
--main_process_ip $MASTER_ADDR \
--main_process_port $MASTER_PORT \
--num_machines 4 \
--machine_rank $NODE_RANK \
--num_processes  32 \
train_wds.py \
--config configs/finetune/imagenet512-latent.yaml \
--resample \
--ckpt_path checkpoints/1050000.pt \
--use_ckpt_path False --use_strict_load False \
--no_amp


