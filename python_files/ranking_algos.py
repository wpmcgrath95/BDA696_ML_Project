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
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
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
        lin_regr_model = statsmodels.api.OLS(y, predictor)
        lin_regr_model_fitted = lin_regr_model.fit()
        print(lin_regr_model_fitted.summary())

        # Get the stats
        t_val = round(lin_regr_model_fitted.tvalues[1], 6)
        p_val = "{:.6e}".format(lin_regr_model_fitted.pvalues[1])

        return t_val, p_val

    # logisitic regression model for a boolean/categorical response
    def logistic_regression_model(self, y, predictor):
        log_regr_model = statsmodels.api.Logit(y, predictor)
        log_regr_model_fitted = log_regr_model.fit()
        print(log_regr_model_fitted.summary())

        # Get the stats
        t_val = round(log_regr_model_fitted.tvalues[1], 6)
        p_val = "{:.6e}".format(log_regr_model_fitted.pvalues[1])

        return t_val, p_val

    # random forest model importance (classifier and regressor)
    # @classmethod
    def random_forest_model(self, X, y, y_type):
        X_train, X_test, y_train, y_test = train_test_split(
            X, np.ravel(y), test_size=0.2, random_state=0
        )

        if y_type == "binary":
            rf_model = RandomForestClassifier(random_state=123)
            rf_model_fitted = rf_model.fit(X_train, y_train)

        else:
            rf_model = RandomForestRegressor(n_estimators=100, random_state=123)
            rf_model_fitted = rf_model.fit(X_train, y_train)

        return rf_model_fitted

    # continous response and continuous predictor plot
    def plot_cont_resp_cont_pred(self, feat, y, **kwargs):
        response = y.columns.to_list()[0]
        response_name = self.dataset[response].name

        stat_text = f'(t-value={kwargs["t_val"]}) (p-value={kwargs["p_val"]})'
        title_text = "Continuous Response by Continuous Predictor"
        fig = px.scatter(x=feat, y=y, trendline="ols")
        fig.update_layout(
            title=f"{title_text}: {stat_text}",
            xaxis_title=f"Predictor: {feat.name}",
            yaxis_title=f"Response: {response_name}",
        )
        fig.show()
        fig.write_html(
            file=f"plots/{feat.name}_scatter_plot.html", include_plotlyjs="cdn"
        )

        return None

    # continous response and categorical predictor plot
    def plot_cont_resp_cat_pred(self, feat, y, **kwargs):
        n = 200
        response = y.columns.to_list()[0]
        response_name = self.dataset[response].name

        # add noise to data
        group_labels = [f"group_{int(i)}" for i in feat.unique()]
        ele_group = pd.cut(feat.to_list(), bins=len(group_labels), labels=group_labels)
        temp_df = pd.DataFrame({"a": feat.values, "b": ele_group})
        temp_df["noise"] = temp_df["a"].values + np.random.normal(
            0, 1, len(temp_df["a"])
        )
        temp_df = temp_df.groupby("b")["noise"].apply(list).reset_index(name="agg")
        group_list = temp_df["agg"].to_list()
        del temp_df

        # Create distribution plot with custom bin_size
        stat_text = f'(t-value={kwargs["t_val"]}) (p-value={kwargs["p_val"]})'
        title_text = "Continuous Response by Categorical Predictor"
        fig_1 = ff.create_distplot(group_list, group_labels, bin_size=2)
        fig_1.update_layout(
            title=f"{title_text}: {stat_text}",
            xaxis_title=f"Response: {response_name}",
            yaxis_title="Distribution",
        )
        fig_1.show()
        fig_1.write_html(
            file=f"plots/{feat.name}_distr_cont_resp_plot.html", include_plotlyjs="cdn"
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
            title=f"{title_text}: {stat_text}",
            xaxis_title="Groupings",
            yaxis_title=f"Response: {response_name}",
        )
        fig_2.show()
        fig_2.write_html(
            file=f"plots/{feat.name}_violin_cont_resp_plot.html", include_plotlyjs="cdn"
        )

        return None

    # categorical response and continuous predictor plot
    def plot_cat_resp_cont_pred(self, feat, y, **kwargs):
        n = 200
        response = y.columns.to_list()[0]
        response_list = self.dataset[response].to_list()
        response_name = self.dataset[response].name

        # add noise to data
        group_labels = [f"group_{int(i)}" for i in self.dataset[response].unique()]
        ele_group = pd.cut(response_list, bins=len(group_labels), labels=group_labels)
        temp_df = pd.DataFrame({"a": self.dataset[response], "b": ele_group})
        temp_df["noise"] = temp_df["a"].values + np.random.normal(
            0, 1, len(temp_df["a"])
        )
        temp_df = temp_df.groupby("b")["noise"].apply(list).reset_index(name="agg")
        group_list = temp_df["agg"].to_list()
        del temp_df

        # Create distribution plot with custom bin_size
        stat_text = f'(t-value={kwargs["t_val"]}) (p-value={kwargs["p_val"]})'
        title_text = "Continuous Predictor by Categorical Response"
        fig_1 = ff.create_distplot(group_list, group_labels, bin_size=2)
        fig_1.update_layout(
            title=f"{title_text}: {stat_text}",
            xaxis_title=f"Predictor: {feat.name}",
            yaxis_title="Distribution",
        )
        fig_1.show()
        fig_1.write_html(
            file=f"plots/{feat.name}_distr_cat_resp_plot.html", include_plotlyjs="cdn"
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
            title=f"{title_text}: {stat_text}",
            xaxis_title=f"Response: {response_name}",
            yaxis_title=f"Predictor: {feat.name}",
        )
        fig_2.show()
        fig_2.write_html(
            file=f"plots/{feat.name}_violin_cat_resp_plot.html", include_plotlyjs="cdn"
        )

        return None

    # categorical response and categorical predictor plot
    def plot_cat_resp_cat_pred(self, feat, y, **kwargs):
        response = y.columns.to_list()[0]
        response_name = self.dataset[response].name

        # Create heatmap plot
        conf_matrix = confusion_matrix(feat.values, self.dataset[response])
        stat_text = f'(t-value={kwargs["t_val"]}) (p-value={kwargs["p_val"]})'
        title_text = "Categorical Predictor by Categorical Response"

        fig = go.Figure(data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max()))
        fig.update_layout(
            title=f"{title_text}: {stat_text}",
            xaxis_title=f"Response: {response_name}",
            yaxis_title=f"Predictor: {feat.name}",
        )
        fig.show()
        fig.write_html(
            file=f"plots/{feat.name}_heatmap_cat_resp_plot.html", include_plotlyjs="cdn"
        )

        return None

    def generate_table():
        # return html_data
        pass

    def main(self):
        # inputing dataset and setting seed
        feat_impt = {}  # dict to hold feature_name: feature_importance
        np.random.seed(seed=123)
        X, y = self.get_pred_and_resp()
        y_type = tot(y)

        self.print_heading("Dataset Info")
        for idx, col in enumerate(X):
            feat = self.dataset[col]
            X_type = self.data_type(feat)
            predictor = statsmodels.api.add_constant(feat)

            if y_type == "continuous" and X_type == "continuous":
                t_val, p_val = self.linear_regression_model(y, predictor)
                self.plot_cont_resp_cont_pred(feat, y, t_val=t_val, p_val=p_val)

            elif y_type == "continuous" and X_type == "categorical":
                t_val, p_val = self.linear_regression_model(y, predictor)
                # need better cutting system
                self.plot_cont_resp_cat_pred(feat, y, t_val=t_val, p_val=p_val)

            elif y_type == "continuous" and X_type == "binary":
                t_val, p_val = self.linear_regression_model(y, predictor)
                self.plot_cont_resp_cat_pred(feat, y, t_val=t_val, p_val=p_val)

            elif y_type == "binary" and X_type == "continuous":
                t_val, p_val = self.logistic_regression_model(y, predictor)
                self.plot_cat_resp_cont_pred(feat, y, t_val=t_val, p_val=p_val)

            elif y_type == "binary" and X_type == "categorical":
                t_val, p_val = self.logistic_regression_model(y, predictor)
                self.plot_cat_resp_cat_pred(feat, y, t_val=t_val, p_val=p_val)

            else:
                print(f"{feat.name} is not a binary or continuous data type")

        self.print_heading("Random Forest Feature Importance")
        rf_model = self.random_forest_model(X, y, y_type)

        for feat, impt in zip(X.columns, rf_model.feature_importances_):
            feat_impt[feat] = impt

        rf_impt_df = pd.DataFrame.from_dict(feat_impt, orient="index").rename(
            columns={0: "Gini-importance"}
        )

        # rf_impt_df.sort_values(by='Gini-importance').plot(kind='bar', rot=45)
        print(rf_impt_df)


if __name__ == "__main__":
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
