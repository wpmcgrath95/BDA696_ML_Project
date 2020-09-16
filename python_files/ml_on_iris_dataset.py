#!/usr/bin/env python3
# Iris Machine Learning Assignment 1
# Will McGrath
# September 8, 2020

import os
import sys

import numpy as np
import pandas as pd
from plotly import express as px
from plotly import graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.svm import LinearSVC


def print_heading(title):
    # creates headers to divide outputs
    print("\n")
    print("*" * 80)
    print(title)
    print("*" * 80)

    return None


def main():
    # trick to get the working folder of this file
    this_dir = os.path.dirname(os.path.realpath(__file__))
    file_loc = os.path.join(this_dir, "../data/iris.data")

    # pulls in the Iris dataset from a CSV and creates a dataframe
    print_heading("Pulling in Data and Creating Dataframe")
    data = pd.read_csv(file_loc, header=None)
    data_df = pd.DataFrame(data).rename(
        columns={
            0: "sepal_len_cm",
            1: "sepal_wid_cm",
            2: "petal_len_cm",
            3: "petal_wid_cm",
            4: "class",
        }
    )
    print(data_df)

    # summary statistics with numpy example
    print_heading("Dataframe Summary Statistics")
    print(data_df.describe())
    # assert method is used to show that numpy and describe() get same results
    assert (
        np.mean(data_df["sepal_len_cm"])
        == data_df.describe().loc["mean", "sepal_len_cm"]
    )

    # defines X and y (class = target)
    print_heading("Defining Covariates and Targets")
    covariates = ["sepal_len_cm", "sepal_wid_cm", "petal_len_cm", "petal_wid_cm"]
    iris_plants = list(data_df["class"].unique())
    X_orig = data_df[covariates].values
    y = data_df["class"].values
    print(f"Columns in Covariates: {covariates}")
    print(f"Shape of Covariates: {X_orig.shape}")
    print(f"Targets in Class Column: {iris_plants}")
    print(f"Shape of Class Column: {y.shape}")

    # scatter plots (can also save the direct html text instead of outputting plot)
    # same as bivariate plots
    print_heading("Plots")
    print("In Browser")
    fig = px.scatter_matrix(
        data_df,
        dimensions=["sepal_len_cm", "sepal_wid_cm", "petal_len_cm", "petal_wid_cm"],
        color="class",
    )
    fig.show()

    # violin plot
    fig = go.Figure()
    for plant in iris_plants:
        fig.add_trace(
            go.Violin(
                y=data_df["sepal_len_cm"][data_df["class"] == plant],
                name=plant + " Sepal Length",
                box_visible=True,
                meanline_visible=True,
            )
        )

        fig.add_trace(
            go.Violin(
                y=data_df["sepal_wid_cm"][data_df["class"] == plant],
                name=plant + " Sepal Width",
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig.show()

    # histograms plots
    fig = px.histogram(data_df, x="petal_len_cm", color="class")
    fig.show()

    fig = px.histogram(data_df, x="petal_wid_cm", color="class")
    fig.show()

    # strip plot
    fig = px.strip(data_df, x="petal_len_cm", y="class", orientation="h", color="class")
    fig.show()

    # parallel coordinates plot
    # note: ['Iris-setosa': 0, 'Iris-versicolor':1, 'Iris-virginica':2]
    plant_names, plant_indx = np.unique(data_df["class"], return_inverse=True)
    fig = px.parallel_coordinates(
        data_df,
        color=plant_indx,
        labels={
            "color": "Plant",
            "sepal_wid_cm": "Sepal Width",
            "sepal_leng_cm": "Sepal Length",
            "petal_wid_cm": "Petal Width",
            "petal_len_cm": "Petal Length",
        },
        color_continuous_scale=px.colors.diverging.Tealrose,
        color_continuous_midpoint=1,
    )

    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Plant", tickvals=list(set(plant_indx)), ticktext=plant_names
        )
    )
    fig.show()

    # adds pipeline
    print_heading("Creating Pipelines")
    rf_pipeline = Pipeline(
        steps=[
            ("Normalizer", Normalizer()),
            ("RandomForest", RandomForestClassifier(random_state=0)),
        ]
    )

    svc_pipeline = Pipeline(
        steps=[
            ("Normalizer", Normalizer()),
            ("OneHotEncoder", OneHotEncoder()),
            ("OneVsRestClassifer", LinearSVC(random_state=0)),
        ]
    )
    print(rf_pipeline)
    print(svc_pipeline)

    print_heading("Pipeline Peformance")
    rf_pipeline.fit(X_orig, y)
    svc_pipeline.fit(X_orig, y)
    rf_probs = rf_pipeline.predict_proba(X_orig)
    rf_preds = rf_pipeline.predict(X_orig)
    svc_preds = svc_pipeline.predict(X_orig)

    print("Overall Performance")
    print("RandomForest Log-Loss:  %.3f" % log_loss(y, rf_probs))
    print("RandomForest Score: %.3f" % rf_pipeline.score(X_orig, y))
    print("LinearSVC Score: %.3f" % svc_pipeline.score(X_orig, y))
    print("", end="\n")

    # encoded target col and preds to get area under ROC Curve
    y_trans_encoded = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))
    svc_preds_trans = OneHotEncoder(sparse=False).fit_transform(
        svc_preds.reshape(-1, 1)
    )
    rf_preds_trans = OneHotEncoder(sparse=False).fit_transform(rf_preds.reshape(-1, 1))

    # calculates area under ROC curve for each target (3) and prints their performance
    # not adding ROC Curve plots since area is 1 so plots won't be useful/interesting
    print("Each Target's Performance (Area Under ROC Curve)")
    for i in range(0, len(iris_plants)):
        roc = roc_auc_score(
            y_trans_encoded[:, i], rf_preds_trans[:, i], multi_class="ovr"
        )
        print(
            "Area under ROC Curve for %s Using RandomForest: %.3f"
            % (iris_plants[i], roc)
        )

        roc = roc_auc_score(
            y_trans_encoded[:, i], svc_preds_trans[:, i], multi_class="ovr"
        )
        print(
            "Area under ROC Curve for %s Using LinearSVC: %.3f" % (iris_plants[i], roc)
        )

    return None


# calls the function main() then exits giving the system the return code of main().
if __name__ == "__main__":
    sys.exit(main())
