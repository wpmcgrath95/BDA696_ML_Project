#!/usr/bin/env python3
# Assignment 4
# Will McGrath

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

import sys

import numpy as np
import pandas as pd
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn import datasets  # metrics
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import type_of_target as tot


class RankingAlgorithms(object):
    def __init__(self, data_file=None):
        if data_file:
            self.dataset = pd.read_csv(data_file)

        else:
            load_data = datasets.load_boston()
            self.dataset = pd.DataFrame(
                data=load_data.data, columns=load_data.feature_names
            )
            self.dataset["target"] = load_data.target

    @staticmethod
    def print_heading(title):
        # creates headers to divide outputs
        print("\n")
        print("*" * 80)
        print(title)
        print("*" * 80)

        return None

    # determines predictors (X) and response (y) cols
    def get_pred_and_resp(self):
        text = "Please enter which column name you would like to use as your response"
        self.print_heading("Choosing predictors and reponse variables")

        while True:
            print(f"Columns: {self.dataset.columns.to_list()}")
            try:
                response = [input(f"{text}: ")]
                predictors = [_ for _ in self.dataset.columns if _ not in response]
                X = self.dataset[predictors]
                y = self.dataset[response]
                break

            except KeyError:
                print("Please enter in a valid column as your response")

        return X, y

    # determining column data type
    def data_type(self, feat):
        X_type = feat.convert_dtypes().dtypes
        print(f"{feat.name} before: {X_type}")

        if X_type == np.float64:
            X_type = "continuous"
        elif type(X_type) == pd.Int64Dtype and np.array_equal(feat.unique(), [0, 1]):
            X_type = "binary"
        else:
            X_type = "categorical"
        print(f"{feat.name} after: {X_type}")

        return X_type

    # linear regression model for a continuous response
    def linear_regression_model(self, y, predictor):
        linear_regression_model = statsmodels.api.OLS(y, predictor)
        linear_regression_model_fitted = linear_regression_model.fit()
        print(linear_regression_model_fitted.summary())

        # Get the stats
        t_val = round(linear_regression_model_fitted.tvalues[1], 6)
        p_val = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

        return t_val, p_val, linear_regression_model_fitted

    # logisitic regression model for a boolean/categorical response
    def logistic_regression_model(self, y, predictor):
        log_reg_model = LogisticRegression(random_state=123).fit(predictor, y)

        return log_reg_model

    # continous response and continuous predictor plot
    def plot_cont_resp_cont_pred(self, feat, y, **kwargs):
        fig = px.scatter(x=feat, y=y, trendline="ols")
        fig.update_layout(
            title=f'Variable: {feat.name}: (t-value={kwargs["t_val"]}) \
                                           (p-value={kwargs["p_val"]})',
            xaxis_title=f"Variable: {feat.name}",
            yaxis_title="y",
        )
        fig.show()
        fig.write_html(
            file=f"plots/{feat.name}_col_scatter_plot.html", include_plotlyjs="cdn"
        )

        return None

    # continous response and categorical predictor plot
    def plot_cont_resp_cat_pred(self, feat, y, **kwargs):
        n = 200

        # add noise to data
        group_labels = [f"group_{i}" for i in feat.unique()]
        ele_group = pd.cut(feat.to_list(), bins=len(group_labels), labels=group_labels)
        temp_df = pd.DataFrame(
            {"a": feat.values, "b": ele_group, "y": np.ravel(y.values)}
        )
        temp_df["noise"] = temp_df["a"].values + np.random.normal(
            0, 1, len(temp_df["a"])
        )
        temp_df = temp_df.groupby("b")["noise"].apply(list).reset_index(name="agg")

        group_list = temp_df["agg"].to_list()
        del temp_df

        # Create distribution plot with custom bin_size
        fig_1 = ff.create_distplot(group_list, group_labels, bin_size=2)
        fig_1.update_layout(
            title=f'Variable: {feat.name}: (t-value={kwargs["t_val"]}) \
                                             (p-value={kwargs["p_val"]})',
            xaxis_title=f"Variable: {feat.name}",
            yaxis_title="Distribution",
        )
        fig_1.show()
        fig_1.write_html(
            file=f"plots/{feat.name}_col_distr_plot.html", include_plotlyjs="cdn"
        )

        fig_2 = go.Figure()
        for curr_hist, curr_group in zip(group_list, group_labels):
            fig_2.add_trace(
                go.Violin(
                    x=np.repeat(curr_group, n),
                    y=curr_hist,
                    name=curr_group,
                    box_visible=True,
                    meanline_visible=True,
                )
            )
        fig_2.update_layout(
            title="Continuous Response by Categorical Predictor",
            xaxis_title="Groupings",
            yaxis_title="Response",
        )
        fig_2.show()
        fig_2.write_html(
            file=f"plots/{feat.name}_col_violin_plot.html", include_plotlyjs="cdn"
        )

        return None

    # categorical response and continuous predictor plot
    def plot_cat_resp_cont_pred(self):
        return None

    # categorical response and categorical predictor plot
    def plot_cat_resp_cat_pred(self):
        return None

    def main(self):
        # inputing dataset and setting seed
        np.random.seed(seed=123)
        X, y = self.get_pred_and_resp()
        y_type = tot(y)

        self.print_heading("Dataset Info")
        for idx, col in enumerate(X):
            feat = self.dataset[col]
            X_type = self.data_type(feat)
            predictor = statsmodels.api.add_constant(feat)

            if y_type == "continuous" and X_type == "continuous":
                t_val, p_val, lin_reg_mod = self.linear_regression_model(y, predictor)
                self.plot_cont_resp_cont_pred(feat, y, t_val=t_val, p_val=p_val)

            elif y_type == "continuous" and X_type == "categorical":
                t_val, p_val, lin_reg_mod = self.linear_regression_model(y, predictor)
                # self.plot_cont_resp_cat_pred(feat, y, t_val=t_val, p_val=p_val)

            elif y_type == "continuous" and X_type == "binary":
                t_val, p_val, lin_reg_mod = self.linear_regression_model(y, predictor)
                self.plot_cont_resp_cat_pred(feat, y, t_val=t_val, p_val=p_val)

            elif y_type == "binary" and X_type == "continuous":
                self.plot_cat_resp_cont_pred()

            elif y_type == "binary" and X_type == "categorical":
                self.plot_cat_resp_cat_pred()

            else:
                print(f"{feat.name} is not a binary or continuous column type")


if __name__ == "__main__":
    answers = ["Yes", "yes", "Y", "y"]

    # input dataset or use default dataset
    while True:
        val = input("Would you like to input a dataset CSV file (Y/N)?: ").capitalize()
        if val == "Y":
            try:
                file = str(input("Please enter the location of your CSV file: "))
                sys.exit(RankingAlgorithms(file).main())
            except FileNotFoundError:
                print("Please enter in an existing CSV file location")
            except ValueError:
                print("Please enter in a CSV file")

        elif val == "N":
            break
        else:
            print("Sorry I do not understand that")

    print("No dataset was chosen so the Boston house-prices dataset will be used.")
    sys.exit(RankingAlgorithms().main())
