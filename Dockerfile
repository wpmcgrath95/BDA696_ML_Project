# base image (host OS)
FROM python:3.8.3

# set the working directory in the container
ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

# get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
  && rm -rf /var/lib/apt/lists/*

# get necessary python libraries
COPY requirements.txt .
RUN pip3 install --compile --no-cache-dir --upgrade pip setuptools
RUN pip3 install --compile --no-cache-dir -r requirements.txt

# create an unprivileged user
RUN useradd --system --user-group --shell /sbin/nologin services
 
# switch to the unprivileged user
USER services

# copy over code 
COPY python_files python_files

# command to run on container start
#CMD ["python", "./server.py"]
CMD ["python", "./python_files/test_docker.py"]