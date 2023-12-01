#!/bin/sh
# weight_path="./weights/yolact_edge_mobilenetv2_54_800000_raw.pth" 
# weight_path="./weights/yolact_edge_mobilenetv2_1749_7000_v2.pth"
# weight_path="./weights/yolact_edge_mobilenetv2_389_4678_tuned_v3.pth"
# weight_path="./weights/yolact_edge_mobilenetv2_272_3273_tuned_v4.pth"

# weight_path="./weights/yolact_edge_mobilenetv2_416_5000_v5.pth"
weight_path="./weights/yolact_edge_mobilenetv2_333_6000_v6.pth"



image_path="./sample_images/PXL_20230330_085036313.MP.jp"
# run_mode="pict"
run_mode="vid"

## for picture
if [ $run_mode == "pict" ]; then
  echo "run pict"
  python3 eval.py \
  --trained_model=$weight_path \
  --score_threshold=0.45 \
  --top_k=10 \
  --image=$image_path \
  --config=turnip_mobilenetv2_config \
  --display \
  --cuda=false \
  --disable_tensorrt
fi


## raw config name
## yolact_edge_mobilenetv2_config

## for video
if [ $run_mode == "vid" ]; then
  echo "run video"
  python3 eval.py \
  --trained_model=$weight_path \
  --score_threshold=0.3 \
  --top_k=10 \
  --config=turnip_mobilenetv2_config \
  --display \
  --cuda=true \
  --disable_tensorrt \
  --video 0 \
  --video_multiframe=1 \
  --display_masks=false \
  --display_bboxes=false \
  --display_text=true \
  --display_size=true \
  --display_dot=true \
  --hide_back=false \
  --display_ajuster=false \
  --only_turnip=true \
  --zoom_rate=1.0
fi

