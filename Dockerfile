# base image
FROM mariadb:latest

# set the working directory in the container
ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

# get necessary python libraries
COPY requirements.txt .

# copy over code 
COPY SQL_files SQL_files
COPY baseball.sql . 

RUN docker container exec -i db-container mysql bb_db < baseball.sql -ppass

# command to run on container start
CMD ["sql", "rolling_batting_avg.sql"]