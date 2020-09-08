# Assignment 1: Machine Learning on Iris Dataset

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

- Clone repo: go to terminal and type `git clone https://github.com/wpmcgrath95/BDA696_ML_Project.git`
  - **Note**: Make sure you are in an empty directory, otherwise type `mkdir newdir`
- Go into directory: `cd newdir`
- Run script: `./scripts/run-ml-code.sh`
  - **Note**: If unable to run script, try typing `chmod +x ./scripts/run-ml-code.sh` and then  
     `./scripts/run-ml-code.sh`
