import numpy as np
import pandas as pd
from plotly import express as px
from plotly import graph_objects as go

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
    np.mean(data_df["sepal_len_cm"]) == data_df.describe().loc["mean", "sepal_len_cm"]
)

# scatter plot
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
