#!/usr/bin/env bash
pip3 install --upgrade pip

pip3 install -r requirements.dev.txt
pip3 install -r requirements.txt
pre-commit install
pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade

docker-compose up db_service
docker container exec -i db-container mysql bbdb < baseball.sql -ppass