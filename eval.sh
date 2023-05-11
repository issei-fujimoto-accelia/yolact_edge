weight_path="./weights/yolact_edge_mobilenetv2_3499_14000.pth"
image_path="./sample_images/PXL_20230330_085036313.MP.jp"
mode = "pic"
# mode = "vid"

## for picture
if $mode = "pic"; then
  python3 eval.py \
  --trained_model=$weight_path \
  --score_threshold=0.45 \
  --top_k=3 \
  --image=$image_path \
  --config=turnip_mobilenetv2_config \
  --display \
  --cuda=true \
  --disable_tensorrt
fi

## for video
if $mode = "vid"; then
  python3 eval.py \
  --trained_model=$weight_path \
  --score_threshold=0.45 \
  --top_k=3 \
  --config=turnip_mobilenetv2_config \
  --display \
  --cuda=true \
  --disable_tensorrt \
  --video 0
fi

