#!/bin/bash

SOURCE_CODE="/home/accelia/i.fujimoto/yolact_edge"

# xhost +localhost
xhost +
docker run --rm --gpus all -it --name=yolact_edge \
  --shm-size=8gb \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $SOURCE_CODE:/yolact_edge/:rw \
  --device /dev/video2:/dev/video0:mwr \
  -e DISPLAY=$DISPLAY \
  yolact_edge_image \
  python /yolact_edge/check_cam.py  
  # bash /yolact_edge/eval.sh
#   
#   bash /yolact_edge/eval.sh
# python /yolact_edge/check_cam.py  




  # bash /yolact_edge/gomi.sh
  # bash /yolact_edge/eval.sh
  # python /yolact_edge/check_cam.py  

