SHOW ERRORS;
SHOW WARNINGS;
COMMIT;

/* 
use baseball schema 
Note: BA = avg(hit/atBat)
*/
USE baseball;

/*
Table: historical table (BA overall) 
Note: COUNT(batter) = number of times batter played and number of games (batter shows up once per game)
*/
DROP TABLE IF EXISTS BATTERS_HISTORICAL;
CREATE TABLE BATTERS_HISTORICAL AS
(SELECT batter AS Batter, COUNT(batter) AS Batters_Count, SUM(Hit) AS Hit_Sum, SUM(atBat) AS atBat_Sum, SUM(Hit)/NULLIF(SUM(atBat), 0) AS Batting_AVG
FROM baseball.batter_counts GROUP BY Batter);

SELECT *
FROM baseball.BATTERS_HISTORICAL;

/*
Table: annual table (BA annually) 
*/
DROP TABLE IF EXISTS BATTERS_ANNUALLY;
CREATE TABLE BATTERS_ANNUALLY AS
(SELECT baseball.batter_counts.batter AS Batter, REPLACE(YEAR(baseball.game.local_date),',','') AS The_Year, COUNT(baseball.batter_counts.batter) AS Batter_Count, 
SUM(baseball.batter_counts.Hit) AS Hit_Sum, SUM(baseball.batter_counts.atBat) AS atBat_Sum, SUM(baseball.batter_counts.Hit)/NULLIF(SUM(baseball.batter_counts.atBat), 0) AS Batting_AVG
FROM baseball.batter_counts
JOIN game ON baseball.batter_counts.game_id = baseball.game.game_id
GROUP BY batter, the_year); 

SELECT *
FROM baseball.BATTERS_ANNUALLY;

/*
Table: rolling table (BA over last 100 days) 
*/
DROP TABLE IF EXISTS BATTERS_ROLLING;
CREATE TABLE BATTERS_ROLLING AS
(SELECT baseball.game.game_id, baseball.batter_counts.batter, DATE(baseball.game.local_date) AS the_date, SUM(baseball.batter_counts.Hit) AS Hit_Sum,
SUM(baseball.batter_counts.atBat) AS atBat_Sum, SUM(baseball.batter_counts.Hit)/NULLIF(SUM(baseball.batter_counts.atBat), 0) AS Batting_AVG
FROM baseball.game
JOIN batter_counts ON baseball.game.game_id = baseball.batter_counts.game_id 
WHERE DATE(baseball.game.local_date) >= DATE(ADDDATE(baseball.game.local_date, INTERVAL -100 DAY))
GROUP BY baseball.game.game_id, batter); 

SELECT *
FROM baseball.BATTERS_ROLLING; 


