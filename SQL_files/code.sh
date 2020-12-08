#!/bin/sh

sleep 10

# Insert in the raw SQL data
if ! mysql -h db-service -uroot -psecret -e 'use baseball'; then
  mysql -h db-service -uroot -psecret -e "create database baseball;"
  mysql -h db-service -uroot -psecret -D baseball < /data/baseball.sql
fi

# Run your scripts
mysql -h db-service -uroot -psecret baseball < /scripts/baseball_batting_average.sql
mysql -h db-service -uroot -psecret baseball < /scripts/rolling_batting_avg.sql

# Get results
mysql -h db-service -uroot -psecret baseball -e '
  SELECT
        Batter,
        (sum(Hits)/sum(Atbats)) as BattingAvg
    FROM BATTERS_ROLLING
    GROUP BY Batter;' > /results/rolling_batting_avg.txt
