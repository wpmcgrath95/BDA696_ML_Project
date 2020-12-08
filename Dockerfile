# base image
FROM ubuntu

# Install mysql client
RUN apt-get update
RUN apt-get install -y mysql-client

# copy over code
COPY baseball.sql /data/baseball.sql
COPY SQL_files /scripts

# command to run on container start
CMD ["/scripts/code.sh"]