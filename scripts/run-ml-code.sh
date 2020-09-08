#!/usr/bin/env bash
chmod +x ./scripts/create-venv.sh
chmod +x ./scripts/run-tests.sh
chmod +x ./python_files/ml_on_iris_dataset.py

source ./scripts/create-venv.sh
source ./scripts/run-tests.sh
python ./python_files/ml_on_iris_dataset.py