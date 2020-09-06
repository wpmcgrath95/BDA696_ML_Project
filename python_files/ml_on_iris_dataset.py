import sys

import numpy as np
import pandas as pd
from plotly import express as px
from plotly import graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import Normalizer, label_binarize
from sklearn.svm import LinearSVC

# import xgboost as xgb # dl was fine but cant run
# from sklearn.pipeline import Pipeline


def main():
    # pulling in the data
    data = pd.read_csv("./data/iris.data", header=None)
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
    print(data_df.describe())
    assert (
        np.mean(data_df["sepal_len_cm"])
        == data_df.describe().loc["mean", "sepal_len_cm"]
    )

    # scatter plot (can also save the direct html text instead of outputting plot)
    fig = px.scatter(data_df, x="sepal_len_cm", y="sepal_wid_cm")
    fig.show()

    # violin plot with all columns
    fig = go.Figure()
    iris_plants = data_df["class"].unique()
    for plant in iris_plants:
        fig.add_trace(
            go.Violin(
                y=data_df["sepal_len_cm"][data_df["class"] == plant],
                name=plant + " length",
                box_visible=True,
                meanline_visible=True,
            )
        )

        fig.add_trace(
            go.Violin(
                y=data_df["sepal_wid_cm"][data_df["class"] == plant],
                name=plant + " width",
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig.show()

    # bar plots

    # other plot

    # other plot

    # transform data using Normalizer and define X and y (targets)
    covariates = ["sepal_len_cm", "sepal_wid_cm", "petal_len_cm", "petal_wid_cm"]
    X_orig = data_df[covariates].values
    normalizer = Normalizer()
    X_trans = normalizer.fit_transform(X_orig)
    y = data_df["class"].values

    # train/test with RandomForestClassifier
    # if split data then could use criterion='entropy' (loss_function) to measure split
    rf_clf = RandomForestClassifier(max_depth=5, random_state=111)
    rf_clf.fit(X_trans, y)
    y_pred = rf_clf.predict(X_trans)

    # RandomForestClassifier performance
    accuracy = accuracy_score(y, y_pred)
    print("Accuracy using RandomForest: %.3f" % accuracy)

    # trans targets (OneHotEncoder works as well but used usually for mult cols)
    y_trans = label_binarize(y, classes=iris_plants)

    # train/test using multiclassification (OneVsRestClassifier) with SVC
    # support vector cld aka type of SVM with linear kernel
    one_vs_rest_clf = OneVsRestClassifier(LinearSVC(random_state=111))
    one_vs_rest_clf.fit(X_trans, y_trans)
    y_pred = one_vs_rest_clf.predict(X_trans)

    # OneVsRestClassifier preformance
    for i in range(0, len(iris_plants)):
        roc = roc_auc_score(y_trans[:, i], y_pred[:, i], multi_class="ovr")
        print("ROC for %s Using OneVsRestClassifier: %.3f" % (iris_plants[i], roc))

    """
    # add pipeline
    print_heading("Model via Pipeline Predictions")
        pipeline = Pipeline(
            [
                ("OneHotEncode",OneHotEncoder()),
                ("RandomForest",RandomForestClassifier(random_state=111)),
            ]
        )
        pipeline.fit(X_orig,y)
    """
    return None


# call the function main() then exits giving the system the return code of main().
if __name__ == "__main__":
    sys.exit(main())
