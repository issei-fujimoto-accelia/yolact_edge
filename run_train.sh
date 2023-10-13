#!/bin/sh

python train.py \
--config=turnip_mobilenetv2_config \
--resume="./pre_weights/yolact_edge_mobilenetv2_54_800000.pth" \
--start_iter=0 \
--batch_size=4 \
--num_workers=0 \
--lr=0.001 \
--dataset=turnip_dataset \
--save_interval=1000 \
--save_folder="./local_tuned" \
--log_folder="./logs"