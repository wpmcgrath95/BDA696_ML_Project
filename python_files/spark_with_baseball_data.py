import sys

from pyspark.sql import SparkSession


def main():
    # Setup Spark
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    print(spark)


if __name__ == "__main__":
    sys.exit(main())
