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

- Clone repo: open terminal and type `git clone https://github.com/wpmcgrath95/BDA696_ML_Project.git`
  - **Note**: Make sure you are in an empty directory, otherwise type `mkdir newdir`
- Go into the directory where the repo is cloned: `cd newdir`
- Run script: `./scripts/run-ml-code.sh`
  - **Note**: If you are unable to run script, try typing `chmod +x ./scripts/run-ml-code.sh` and then  
     `./scripts/run-ml-code.sh`
