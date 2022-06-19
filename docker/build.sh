#!/usr/bin/bash

tag=${1:-"1.2"}

echo "docker image 'dm_control:$tag'"

docker build -t dm_control:$tag .
