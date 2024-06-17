#!/bin/bash
cd $(dirname $0)/../docker
export COMPOSE_PROJECT_NAME="fastc-server"
docker compose up --build --build # -d --build
