# Machine Learning on Iris Dataset

## Objectives

- Download [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) and calculate simple summary statistics
- Plot the different classes (targets) using 5 different plots
- Analyze and build models using scikit-learn
  - **Multiclassification problem**
  - No train/test split
- Calculate performance using different metrics
- Create a shell script that will run code for any user

## Setup for Developement:

- Setup a python 3.x venv (usually in `.venv`)
  - You can run `./scripts/create-venv.sh` to generate one
- `pip3 install --upgrade pip`
- Install dev requirements `pip3 install -r requirements.dev.txt`
- Install requirements `pip3 install -r requirements.txt`
- `pre-commit install`

### Update Versions

`pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade`

### Run `pre-commit` Locally.

`pre-commit run --all-files`

## Setup for Users

- Make sure you are in an empty directory, otherwise create one: `mkdir newdir`
  - **Note**, enter into the directory where the repo will be cloned: `cd newdir`
- Clone repo: `git clone -b Assignment_1 https://github.com/wpmcgrath95/BDA696_ML_Project.git`
- Enter into the BDA696_ML_Project directory: `cd BDA696_ML_Project`
- Run script for Iris data: `./scripts/run-ml-code.sh`
  - **Note**, if you're unable to run the script, try:
    ```bash
    chmod +x ./scripts/run-ml-code.sh
    ./scripts/run-ml-code.sh
    ```
- Run script for assignment 4: `./scripts/run-ranking-algos.sh`
