DROP TABLE IF EXISTS BATTERS_ROLLING;
CREATE TABLE BATTERS_ROLLING AS
    SELECT 
            bc.batter, 
            g.game_id,
            bc.Hit, 
            bc.atBat, 
            g.local_date,
            SUM(bc.atBat) AS Atbats,
            SUM(bc.Hit) as Hits

    FROM baseball.batter_counts bc
    JOIN baseball.game g ON g.game_id = bc.game_id
    JOIN baseball.batter_counts bc1 ON bc.batter = bc1.batter
	JOIN baseball.game g1 ON g1.game_id = bc1.game_id AND
        g.local_date >= DATE(ADDDATE(g.local_date, INTERVAL -100 DAY))
    WHERE g.local_date < '2011-04-04 15:05:00' AND bc.batter IN (SELECT Batter FROM batter_counts bc2 WHERE game_id = '12560')
	GROUP BY Batter, g.game_id
	ORDER BY Batter;
