accelerate launch \
--main_process_ip $MASTER_ADDR \
--main_process_port $MASTER_PORT \
--num_machines 4 \
--machine_rank $NODE_RANK \
--num_processes  32 \
train_wds.py \
--config configs/train/imagenet512-latent.yaml \
--resample


