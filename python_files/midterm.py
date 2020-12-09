#!/usr/bin/env python3
# Will McGrath
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api
from create_html_table import CreateHTMLTable
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn import datasets
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import type_of_target as tot


class Midterm(object):
    def __init__(self, data_file=None):
        if data_file:
            self.dataset = pd.read_csv(data_file)
            self.dataset = self.dataset.dropna()

        else:
            load_data = datasets.load_boston()
            self.dataset = pd.DataFrame(
                data=load_data.data, columns=load_data.feature_names
            )
            self.dataset["target"] = load_data.target

    @staticmethod
    def print_heading(title: str) -> str:
        # creates headers to divide outputs
        print("\n")
        print("*" * 90)
        print(title)
        print("*" * 90)

        return None

    # determines predictors (X) and response (y) cols
    def get_pred_and_resp(self):
        text = "Please enter which column name you would like to use as your response"
        self.print_heading("Choosing Predictors and Response Variables")

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
    def data_type(self, feat) -> str:
        X_type = feat.convert_dtypes().dtypes

        if X_type == np.float64:
            X_type = "continuous"
        elif type(X_type) == pd.Int64Dtype and np.array_equal(feat.unique(), [0, 1]):
            X_type = "binary"
        else:
            X_type = "categorical"
        print(f"Predictor: {feat.name} {X_type} data type")

        return X_type

    # linear regression model for a continuous response
    def linear_regression_model(self, y, predictor):
        lin_regr_model = statsmodels.api.OLS(y, predictor)
        lin_regr_model_fitted = lin_regr_model.fit()

        # Get the stats
        t_val = round(lin_regr_model_fitted.tvalues[1], 6)
        p_val = "{:.6e}".format(lin_regr_model_fitted.pvalues[1])

        return t_val, p_val

    # logisitic regression model for a boolean/categorical response
    def logistic_regression_model(self, y, predictor):
        log_regr_model = statsmodels.api.Logit(y, predictor)
        log_regr_model_fitted = log_regr_model.fit()

        # Get the stats
        t_val = round(log_regr_model_fitted.tvalues[1], 6)
        p_val = "{:.6e}".format(log_regr_model_fitted.pvalues[1])

        return t_val, p_val

    def brute_force(self, X, y, y_type):
        if y_type == "binary":
            est = LinearRegression()
            efs = EFS(
                estimator=est,
                min_features=1,
                max_features=2,
                scoring="neg_mean_squared_error",
                cv=5,
            )

            efs = efs.fit(X, y)
            efs_df = pd.DataFrame.from_dict(efs.get_metric_dict()).T
            efs_df.sort_values("avg_score", inplace=True, ascending=False)

        else:
            est = LogisticRegression()
            efs = EFS(
                estimator=est,
                min_features=1,
                max_features=2,
                scoring="neg_mean_squared_error",
                cv=5,
            )

            efs = efs.fit(X, y)
            efs_df = pd.DataFrame.from_dict(efs.get_metric_dict()).T
            efs_df.sort_values("avg_score", inplace=True, ascending=False)

        # horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, 9))
        y_pos = np.arange(len(efs_df))
        ax.barh(y_pos, efs_df["avg_score"], xerr=efs_df["std_dev"])
        ax.set_yticks(y_pos)
        ax.set_xlabel("Avg Score")
        ax.set_ylabel("Feature Names")
        ax.tick_params(labelleft=False)
        plt.show()

        return efs_df

    # continous response and continuous predictor plot
    def plot_cont_resp_cont_pred(self, feat, y, y_name, **kwargs):
        stat_text = f'(t-value={kwargs["t_val"]}) (p-value={kwargs["p_val"]})'
        title_text = "Continuous Response by Continuous Predictor"
        fig = px.scatter(x=feat, y=y, trendline="ols")
        fig.update_layout(
            title=f"{title_text}: {stat_text}",
            xaxis_title=f"Predictor: {feat.name}",
            yaxis_title=f"Response: {y_name}",
        )
        plot_file = f"plots/{feat.name}_scatter_plot.html"
        fig.write_html(file=plot_file, include_plotlyjs="cdn")

        return None

    # continous response and categorical predictor plot
    def plot_cont_resp_cat_pred(self, feat, y, y_name, **kwargs):
        n = 200

        # add noise to data
        group_labels = [f"group_{int(i)}" for i in range(len(feat.unique()))]
        ele_group = pd.cut(feat.to_list(), bins=len(group_labels), labels=group_labels)
        temp_df = pd.DataFrame({"a": feat.values, "b": ele_group})
        temp_df["noise"] = temp_df["a"].values + np.random.normal(
            0, 1, len(temp_df["a"])
        )
        temp_df = temp_df.groupby("b")["noise"].apply(list).reset_index(name="agg")
        temp_df = temp_df[temp_df["agg"].astype(bool)]
        group_list = temp_df["agg"].to_list()
        group_labels = [f"group_{int(i)}" for i in range(1, len(temp_df["agg"]) + 1)]
        del temp_df

        # Create distribution plot with custom bin_size
        stat_text = f'(t-value={kwargs["t_val"]}) (p-value={kwargs["p_val"]})'
        title_text = "Continuous Response by Categorical Predictor"
        fig_1 = ff.create_distplot(group_list, group_labels, bin_size=0.2)
        fig_1.update_layout(
            title=f"{title_text}: {stat_text}",
            xaxis_title=f"Response: {y_name}",
            yaxis_title="Distribution",
        )
        plot_file_1 = f"plots/{feat.name}_distr_cont_resp_plot.html"
        fig_1.write_html(file=plot_file_1, include_plotlyjs="cdn")

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
            yaxis_title=f"Response: {y_name}",
        )
        plot_file_2 = f"plots/{feat.name}_violin_cont_resp_plot.html"
        fig_2.write_html(file=plot_file_2, include_plotlyjs="cdn")

        return None

    # categorical response and continuous predictor plot
    def plot_cat_resp_cont_pred(self, feat, y, y_name, **kwargs):
        n = 200
        response_list = self.dataset[y_name].to_list()

        # add noise to data
        group_labels = [f"group_{int(i)}" for i in self.dataset[y_name].unique()]
        ele_group = pd.cut(response_list, bins=len(group_labels), labels=group_labels)
        temp_df = pd.DataFrame({"a": self.dataset[y_name], "b": ele_group})
        temp_df["noise"] = temp_df["a"].values + np.random.normal(
            0, 1, len(temp_df["a"])
        )
        temp_df = temp_df.groupby("b")["noise"].apply(list).reset_index(name="agg")
        group_list = temp_df["agg"].to_list()
        del temp_df

        # Create distribution plot with custom bin_size
        # USE F-STAT AND P-VALUE FOR F-STAT FOR CATE RESP.
        stat_text = f'(t-value={kwargs["t_val"]}) (p-value={kwargs["p_val"]})'
        title_text = "Continuous Predictor by Categorical Response"
        fig_1 = ff.create_distplot(group_list, group_labels, bin_size=2)
        fig_1.update_layout(
            title=f"{title_text}: {stat_text}",
            xaxis_title=f"Predictor: {feat.name}",
            yaxis_title="Distribution",
        )
        plot_file_1 = f"plots/{feat.name}_distr_cat_resp_plot.html"
        fig_1.write_html(file=plot_file_1, include_plotlyjs="cdn")

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
            xaxis_title=f"Response: {y_name}",
            yaxis_title=f"Predictor: {feat.name}",
        )
        plot_file_2 = f"plots/{feat.name}_violin_cat_resp_plot.html"
        fig_2.write_html(file=plot_file_2, include_plotlyjs="cdn")

        return None

    # categorical response and categorical predictor plot
    def plot_cat_resp_cat_pred(self, feat, y, y_name, **kwargs):
        # Create heatmap plot
        conf_matrix = confusion_matrix(feat.values, self.dataset[y_name])
        stat_text = f'(t-value={kwargs["t_val"]}) (p-value={kwargs["p_val"]})'
        title_text = "Categorical Predictor by Categorical Response"

        fig = go.Figure(data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max()))
        fig.update_layout(
            title=f"{title_text}: {stat_text}",
            xaxis_title=f"Response: {y_name}",
            yaxis_title=f"Predictor: {feat.name}",
        )
        plot_file = f"plots/{feat.name}_heatmap_cat_resp_plot.html"
        fig.write_html(file=plot_file, include_plotlyjs="cdn")

        return None

    def diff_with_mean_response(self, feat, X_type, y_type, y_name):
        if y_type == "continuous" and X_type == "continuous":

            # might want to use histogram
            cut = pd.cut(feat, 10)
            vals = cut.value_counts(dropna=False)
            print(cut.value_counts())
            diff_mean_list = []
            for i in self.dataset[y_name]:
                if i > -0.1 and i <= 10.0:
                    diff_mean_list.append(i)
            print(np.mean(diff_mean_list))

            avg_pop_mean = self.dataset[y_name].mean()
            cut_df = vals.rename_axis("unique_values").reset_index(name="counts")
            cut_df["pop_mean"] = [avg_pop_mean] * len(cut_df)
            cut_df["bin_mean"] = cut_df["unique_values"].apply(
                lambda x: x.astype(str).str.strip("()[]")
            )

            print(cut_df)

        return None

    def knot_points(self):
        # try out knot points take abs value of a var
        pass

    def main(self):
        # trick to get the working folder of this file
        this_dir = os.path.dirname(os.path.realpath(__file__))
        plot_path = os.path.join(this_dir, "../plots")
        html_path = os.path.join(this_dir, "../html_files")

        if not os.path.isdir(plot_path):
            os.mkdir(plot_path)  # create plots directory
            print("Directory '% s' created" % plot_path)

        if not os.path.isdir(html_path):
            os.mkdir(html_path)  # create plots directory
            print("Directory '% s' created" % html_path)

        np.random.seed(seed=1)
        X, y = self.get_pred_and_resp()
        y_name = y.columns.to_list()[0]
        y_type = tot(y)

        self.print_heading("Dataset")
        print(self.dataset)
        print(f"Response: {y_name} {y_type} data type")
        if y_type not in ["continuous", "binary"]:
            print("\n")
            print("Please choose a different response")
            sys.exit(self.main())

        self.print_heading("Dataset Stats")
        info_dict = {}
        feat_list, feat_type_list = [], []
        for idx, col in enumerate(X):
            feat = self.dataset[col]
            X_type = self.data_type(feat)
            predictor = statsmodels.api.add_constant(feat)

            if y_type == "continuous" and X_type == "continuous":
                t_val, p_val = self.linear_regression_model(y, predictor)
                self.plot_cont_resp_cont_pred(feat, y, y_name, t_val=t_val, p_val=p_val)

            elif y_type == "continuous" and X_type == "categorical":
                t_val, p_val = self.linear_regression_model(y, predictor)
                # error with plotting

            elif y_type == "continuous" and X_type == "binary":
                t_val, p_val = self.linear_regression_model(y, predictor)
                self.plot_cont_resp_cat_pred(feat, y, y_name, t_val=t_val, p_val=p_val)

            elif y_type == "binary" and X_type == "continuous":
                t_val, p_val = self.logistic_regression_model(y, predictor)
                self.plot_cat_resp_cont_pred(feat, y, y_name, t_val=t_val, p_val=p_val)

            elif y_type == "binary" and X_type == "categorical":
                t_val, p_val = self.logistic_regression_model(y, predictor)
                self.plot_cat_resp_cat_pred(feat, y, y_name, t_val=t_val, p_val=p_val)

            else:
                print(f"{feat.name} isn't a binary or continuous data type")

            # difference with mean response
            print("Bin Count")
            self.diff_with_mean_response(feat, X_type, y_type, y_name)
            feat_list.append(feat.name)
            feat_type_list.append(X_type)

        # plot correlation metrics
        self.print_heading("Correlation Metrics")
        corr_matrix = self.dataset.corr(method="pearson")

        # get each feat corr metrics
        print(corr_matrix.sort_values(y_name, ascending=False))

        # getting info and all plot paths
        info_dict["response"] = [y_name] * len(feat_list)
        info_dict["predictor"] = feat_list
        info_dict["response_type"] = [y_type] * len(feat_list)
        info_dict["predictor_type"] = feat_type_list
        info_dict["correlation"] = corr_matrix[y_name].to_list()
        all_plot_paths = [os.path.join(plot_path, x) for x in os.listdir(plot_path)]

        # brute force
        # self.print_heading("Brute Force")
        # efs_df = self.brute_force(X, y, y_type)
        # print(efs_df)

        self.print_heading("HTML Table With Plots")
        CreateHTMLTable(info_dict, all_plot_paths).main()


if __name__ == "__main__":
    # input dataset or use default dataset
    while True:
        val = input("Would you like to input a dataset CSV file (Y/N)?: ").capitalize()
        if val == "Y":
            try:
                file = str(input("Please enter the location of your CSV file: "))
                sys.exit(Midterm(file).main())
            except FileNotFoundError:
                print("Please enter in an existing CSV file location")
            except ValueError as err:
                print(err)
                print("Please enter in a CSV file")

        elif val == "N":
            break
        else:
            print("Sorry I don't understand that")

    print("No dataset was chosen so the Boston house-prices dataset will be used")
    sys.exit(Midterm().main())
