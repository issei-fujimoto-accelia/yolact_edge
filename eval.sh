#!/bin/sh
weight_path="./weights/yolact_edge_mobilenetv2_3499_14000.pth"
image_path="./sample_images/PXL_20230330_085036313.MP.jp"
# run_mode="pict"
run_mode="vid"

## for picture
if [ $run_mode == "pict" ]; then
  echo "run pict"
  python3 eval.py \
  --trained_model=$weight_path \
  --score_threshold=0.45 \
  --top_k=3 \
  --image=$image_path \
  --config=turnip_mobilenetv2_config \
  --display \
  --cuda=false \
  --disable_tensorrt
fi

## for video
if [ $run_mode == "vid" ]; then
  echo "run video"
  python3 eval.py \
  --trained_model=$weight_path \
  --score_threshold=0.45 \
  --top_k=3 \
  --config=turnip_mobilenetv2_config \
  --display \
  --cuda=true \
  --disable_tensorrt \
  --video 0 \
  --display_masks=true \
  --display_bboxes=false \
  --display_text=false \
  --display_size=true \
  --hide_back=true
fi

