# Machine Learning Engineering

## Objectives

- Machine Learning assignments from class BDA 696 at SDSU Fall '20

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
- Clone repo: `git clone https://github.com/wpmcgrath95/BDA696_ML_Project.git`
- Enter into the BDA696_ML_Project directory: `cd BDA696_ML_Project`

### Docker Objective

- Use Docker to recreate batting average from a SQL database

## Setup for Docker:

- Extract baseball.sql in root folder
- Run `chmod +x ./scripts/run-docker.sh`
- Then run `./scripts/run-docker.sh`
