#!/usr/bin/bash
set -euo pipefail

project_home="/home/accelia/i.fujimoto/yolact_edge"

# status=`sudo -n $project_home/service/cec_status.sh`

# echo "status: "$status

# if [[ ${status} = "power status: on" ]]; then
#     sudo -n $project_home/service/cec_off.sh
# fi

# if [[ ${status} = "power status: standby" ]]; then
#     sudo -n $project_home/service/cec_start.sh
# fi

# status=`sudo -n $project_home/service/cec_status.sh`
# echo "status: "$status

# if [[ ${status} = "power status: standby" ]]; then
#     # pid=`cat $project_home/log_running_pid`
#     # kill $pid
#     echo "power off success"
#     exit 0
# fi

sudo -n $project_home/service/cec_start.sh
status=`sudo -n $project_home/service/cec_status.sh`
if [[ ${status} = "power status: on" ]]; then
    xrandr --output HDMI-1-0 --auto
    exec $project_home/.yolact_edge/bin/python3 $project_home/run_with_window.py
    # pid=$!
    # echo $pid > $project_home/log_running_pid
    # exit 0
fi
