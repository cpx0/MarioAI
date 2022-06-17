#!/usr/bin/bash

NB_USER=cpx0
CMDNAME=`basename $0`

echo "Usage: $CMDNAME container_name image_tag" 1>&2

echo "JupyterLab open at http://127.0.0.1:9280 "

container=${1:-"dm_control"}
tag=${2:-"1.1"}

echo "docker container name: $container"
echo "docker image 'dm_control:$tag'"

docker run --gpus all -ti --rm --name $container -p 9280:2980 -v $(pwd):/home/$NB_USER/workspace -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix:10.0 dm_control:$tag
