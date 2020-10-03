# Using Spark With Baseball DB Assignment 3
# Will McGrath
# October 3, 2020

import os
import sys

from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer
from pyspark.sql import SparkSession
from split_column_transform import SplitColumnTransform


def main():
    # Setup Spark and fixing time-zone issue
    SUBMIT_ARGS = "--packages mysql:mysql-connector-java:8.0.11 pyspark-shell"
    END_ARG = "?zeroDateTimeBehavior=CONVERT_TO_NULL&useUnicode=true& \
                useJDBCCompliantTimezoneShift=true& \
                useLegacyDatetimeCode=false&serverTimezone=PST"
    os.environ["PYSPARK_SUBMIT_ARGS"] = SUBMIT_ARGS

    # Spark connection
    spark = (
        SparkSession.builder.master("local[*]")
        .config("spark.sql.debug.maxToStringFields", 100)
        .getOrCreate()
    )
    database = "baseball"
    port = "3306"
    user = "root"
    password = ""

    # create batter_count table
    batter_df = (
        spark.read.format("jdbc")
        .options(
            url=f"jdbc:mysql://localhost:{port}/{database}{END_ARG}",
            driver="com.mysql.cj.jdbc.Driver",
            dbtable="batter_counts",
            user=user,
            password=password,
        )
        .load()
    )

    batter_df.createOrReplaceTempView("batter_counts")
    batter_df.persist(StorageLevel.DISK_ONLY)

    # create game table
    game_df = (
        spark.read.format("jdbc")
        .options(
            url=f"jdbc:mysql://localhost:{port}/{database}{END_ARG}",
            driver="com.mysql.cj.jdbc.Driver",
            dbtable="game",
            user=user,
            password=password,
        )
        .load()
    )

    game_df.createOrReplaceTempView("game")
    game_df.persist(StorageLevel.DISK_ONLY)

    batting_avg_df = spark.sql(
        """
        SELECT batter_counts.batter AS Batter,
               REPLACE(YEAR(game.local_date),',','') AS The_Year,
               COUNT(batter_counts.batter) AS Batter_Count,
               SUM(batter_counts.Hit) AS Hit_Sum,
               SUM(batter_counts.atBat) AS atBat_Sum,
               SUM(batter_counts.Hit)/NULLIF(SUM(batter_counts.atBat), 0) AS Batting_AVG
        FROM batter_counts
        JOIN game ON batter_counts.game_id = game.game_id
        GROUP BY Batter, The_Year
        """
    )

    # using transformer
    split_column_transform = SplitColumnTransform(
        inputCols=["The_Year"], outputCol="categorical"
    )

    count_vectorizer = CountVectorizer(
        inputCol="categorical", outputCol="categorical_vector"
    )

    pipeline = Pipeline(stages=[split_column_transform, count_vectorizer])

    model = pipeline.fit(batting_avg_df)
    batting_avg_df = model.transform(batting_avg_df)
    batting_avg_df.show()


if __name__ == "__main__":
    sys.exit(main())
