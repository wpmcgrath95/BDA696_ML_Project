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
(SELECT baseball.batter_counts.batter AS Batter, REPLACE(YEAR(baseball.game.local_date),',','') AS The_Year, 
COUNT(baseball.batter_counts.batter) AS Batter_Count, SUM(baseball.batter_counts.Hit) AS Hit_Sum, SUM(baseball.batter_counts.atBat) AS atBat_Sum,
SUM(baseball.batter_counts.Hit)/NULLIF(SUM(baseball.batter_counts.atBat), 0) AS Batting_AVG
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
(SELECT baseball.batter_counts.batter, baseball.batter_counts.Hit, baseball.batter_counts.atBat, baseball.game.local_date
FROM baseball.batter_counts 
JOIN game on baseball.batter_counts.game_id = baseball.game.game_id); 

SELECT DATE(a.local_date) AS The_Date, DATE(ADDDATE(a.local_date, INTERVAL -100 DAY)) as Date_100_Days_Before, a.batter AS Batter,
   (SELECT SUM(b.Hit)/NULLIF(SUM(b.atBat), 0)
	FROM baseball.BATTERS_ROLLING AS b
	WHERE DATEDIFF(a.local_date, b.local_date) BETWEEN 0 AND 100 AND b.batter = a.batter) AS Rolling_Window_Batting_AVG_100_Days
FROM baseball.BATTERS_ROLLING AS a
ORDER BY a.local_date ASC 
LIMIT 0,100; 

/*
SELECT a.local_date, SUM(a.Hit)/nullif (SUM(a.atBat),0)
FROM baseball.BATTERS_ROLLING AS a
JOIN baseball.BATTERS_ROLLING AS b
ON a.local_date WHERE DATEDIFF(a.local_date,b.local_date) BETWEEN 0 AND 100 AND a.batter=b.batter
ORDER BY a.local_date ASC 
LIMIT 0,20; # use join on itself instead of 2nd select
*/