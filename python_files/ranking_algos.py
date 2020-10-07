# Assignment 4
# Will McGrath
# October 6, 2020

"""
Input: random dataframe

Description:
1. Determine variable type for covariates (predictor) and target (response)
2. Calculate the different ranking algos using:
    - p-value and t-score along with its plot
    - Difference with mean of response along with its plot (weighted and unweighted)
    - RandomForest Variable importance ranking
3. Generate a table with all the variables and their rankings

Output: HTML based rankings report with links to the plots
    - table with all the variables and their rankings
"""

"""
import sys
from datetime import datetime

import numpy as np
import statsmodels.api
from plotly import express as px
from random_data_generator import RandomDataGenerator
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
"""


class RankingAlgorithms(object):
    def __init__(self, data_file=None):
        pass

    def main(self):
        pass


if __name__ == "__main__":
    pass
