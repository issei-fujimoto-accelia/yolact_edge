#!/bin/sh
# weight_path="./weights/yolact_edge_mobilenetv2_54_800000_raw.pth" 
# weight_path="./weights/yolact_edge_mobilenetv2_1749_7000_v2.pth"
# weight_path="./weights/yolact_edge_mobilenetv2_389_4678_tuned_v3.pth"
# weight_path="./weights/yolact_edge_mobilenetv2_272_3273_tuned_v4.pth"

# weight_path="./weights/yolact_edge_mobilenetv2_416_5000_v5.pth" ## OK (前回まで使ってたやつ)
# weight_path="./weights/yolact_edge_mobilenetv2_333_6000_v6.pth"

# weight_path="./weights/yolact_edge_resnet101_99_2000_v7_1.pth" ## 重なりあり、手なし (これが良さそう)
# weight_path="./weights/yolact_edge_resnet101_361_3255_v7_2.pth" ## 重なりあり、手あり
weight_path="./weights/yolact_edge_resnet101_244_1711_v8.pth" ## 重なりなし、手なし (これが良さそう)


## for video sample
# weight_path="./raw_weights/yolact_edge_vid_847_50000.pth"

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

# 30 ~ (50あれば余裕)
## raw config name
## yolact_edge_mobilenetv2_config

# --config=turnip_mobilenetv2_config \  
# --config=yolact_edge_resnet50_config \
# --config=turnip_restnet101_config \
## for video
if [ $run_mode == "vid" ]; then
  echo "run video"
  python3 eval.py \
  --score_threshold=0.6 \
  --trained_model=$weight_path \
  --top_k=100 \
  --config=turnip_restnet101_config \
  --display \
  --cuda=true \
  --disable_tensorrt \
  --video 2 \
  --video_multiframe=1 \
  --display_masks=true \
  --display_bboxes=false \
  --display_text=true \
  --display_size=true \
  --display_dot=false \
  --display_scores=false \
  --hide_back=false  \
  --display_ajuster=false \
  --only_turnip=true \
  --zoom_rate=1.00 \
  --use_fp16_tensorrt
fi

