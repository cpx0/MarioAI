#!/usr/bin/bash

USER_NAME=cpx0
CMDNAME=`basename $0`


function main() {

    echo "Usage: $CMDNAME container_name host_port image_tag"

    container=${1:-"mario_ai"}
    host_port=${2:-"9280"}
    tag=${3:-"1.2"}

    echo "docker container name: $container"
    echo "docker host port for JupyterLab: $host_port"
    echo "docker image 'dm_control:$tag'"

    echo "JupyterLab open at http://127.0.0.1:$host_port "

    docker run --gpus all -ti --rm --hostname $(hostname) --name $container \
        -p $host_port:2980 -v $(pwd):/home/$USER_NAME/workspace \
        -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
        dm_control:$tag

    # docker-compose up --no-recreate

}

main "$@"

