# import all needed packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import plotly.express as px
import chart_studio as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# import the rental dataset using pandas
rent_df = pd.read_csv("/Users/izzyabate/Downloads/archive (1)/House_Rent_Dataset.csv")

# check the dataset for null and duplicated values
rent_df.isnull().sum()
rent_df.duplicated().sum()

# Exploring the dataset
print(rent_df.head())
print(rent_df.describe())
print(rent_df.columns)

# Visualize the correlation your numerical data
rent_pairs = sns.pairplot(data=rent_df, kind="kde")
rent_pairs.fig.set_size_inches(15, 15)
rent_pairs.fig.suptitle("Pair Plot", y=1.02)
plt.show()

# Bivariate Analysis of variables using Box plots
rent_fig = make_subplots(
    rows=7,
    cols=2,
    subplot_titles=(
        "BHK vs Rent",
        "BHK vs Size",
        "Area Type vs Rent",
        "Area Type vs Size",
        "City vs Rent",
        "City vs Size",
        "Furnishing Status vs Rent",
        "Furnishing Status vs Size",
        "Tenant Preferred vs Rent",
        "Tenant Preferred vs Size",
        "Bathroom vs Size",
        "Bathroom vs Rent",
        "Point of Contact vs Size",
        "Point of Contact vs Rent",
    ),
)
rent_fig.append_trace(
    go.Box(
        x=rent_df["BHK"].values,
        y=rent_df["Rent"].values,
        name="BHK vs Rent",
        boxmean="sd",
    ),
    row=1,
    col=1,
)

rent_fig.append_trace(
    go.Box(
        x=rent_df["BHK"].values,
        y=rent_df["Size"].values,
        name="BHK vs Size",
        boxmean="sd",
    ),
    row=1,
    col=2,
)

# Row 2
rent_fig.append_trace(
    go.Box(
        x=rent_df["Area Type"].values,
        y=rent_df["Rent"].values,
        name="Area Type vs Rent",
        boxmean="sd",
    ),
    row=2,
    col=1,
)

rent_fig.append_trace(
    go.Box(
        x=rent_df["Area Type"].values,
        y=rent_df["Size"].values,
        name="Area Type vs Size",
        boxmean="sd",
    ),
    row=2,
    col=2,
)

# Row 3
rent_fig.append_trace(
    go.Box(
        x=rent_df["City"].values,
        y=rent_df["Rent"].values,
        name="City vs Rent",
        boxmean="sd",
    ),
    row=3,
    col=1,
)
rent_fig.append_trace(
    go.Box(
        x=rent_df["City"].values,
        y=rent_df["Size"].values,
        name="City vs Size",
        boxmean="sd",
    ),
    row=3,
    col=2,
)
rent_fig.append_trace(
    go.Box(
        x=rent_df["Furnishing Status"].values,
        y=rent_df["Rent"].values,
        name="Furnishing Status vs Rent",
        boxmean="sd",
    ),
    row=4,
    col=1,
)
rent_fig.append_trace(
    go.Box(
        x=rent_df["Furnishing Status"].values,
        y=rent_df["Size"].values,
        name="Furnishing Status vs Size",
        boxmean="sd",
    ),
    row=4,
    col=2,
)
rent_fig.append_trace(
    go.Box(
        x=rent_df["Tenant Preferred"].values,
        y=rent_df["Rent"].values,
        name="Tenant Preferred vs Rent",
        boxmean="sd",
    ),
    row=5,
    col=1,
)
rent_fig.append_trace(
    go.Box(
        x=rent_df["Tenant Preferred"].values,
        y=rent_df["Size"].values,
        name="Tenant Preferred vs Size",
        boxmean="sd",
    ),
    row=5,
    col=2,
)
rent_fig.append_trace(
    go.Box(
        x=rent_df["Bathroom"].values,
        y=rent_df["Rent"].values,
        name="Bathroom vs Rent",
        boxmean="sd",
    ),
    row=6,
    col=1,
)
rent_fig.append_trace(
    go.Box(
        x=rent_df["Bathroom"].values,
        y=rent_df["Size"].values,
        name="Bathroom vs Size",
        boxmean="sd",
    ),
    row=6,
    col=2,
)
rent_fig.append_trace(
    go.Box(
        x=rent_df["Point of Contact"].values,
        y=rent_df["Rent"].values,
        name="Point of Contact vs Rent",
        boxmean="sd",
    ),
    row=7,
    col=1,
)
rent_fig.append_trace(
    go.Box(
        x=rent_df["Point of Contact"].values,
        y=rent_df["Size"].values,
        name="Point of Contact vs Size",
        boxmean="sd",
    ),
    row=7,
    col=2,
)
rent_fig["layout"]["xaxis"]["title"] = "BHK"
rent_fig["layout"]["xaxis2"]["title"] = "BHK"
rent_fig["layout"]["yaxis"]["title"] = "Rent"
rent_fig["layout"]["yaxis2"]["title"] = "Size"

rent_fig["layout"]["xaxis3"]["title"] = "Area Type"
rent_fig["layout"]["xaxis4"]["title"] = "Area Type"
rent_fig["layout"]["yaxis3"]["title"] = "Rent"
rent_fig["layout"]["yaxis4"]["title"] = "Size"

rent_fig["layout"]["xaxis5"]["title"] = "City"
rent_fig["layout"]["xaxis6"]["title"] = "City"
rent_fig["layout"]["yaxis5"]["title"] = "Rent"
rent_fig["layout"]["yaxis6"]["title"] = "Size"

rent_fig["layout"]["xaxis7"]["title"] = "Furnishing Status"
rent_fig["layout"]["xaxis8"]["title"] = "Furnishing Status"
rent_fig["layout"]["yaxis7"]["title"] = "Rent"
rent_fig["layout"]["yaxis8"]["title"] = "Size"

rent_fig["layout"]["xaxis9"]["title"] = "Tenant Preferred"
rent_fig["layout"]["xaxis10"]["title"] = "Tenant Preferred"
rent_fig["layout"]["yaxis9"]["title"] = "Rent"
rent_fig["layout"]["yaxis10"]["title"] = "Size"

rent_fig["layout"]["xaxis11"]["title"] = "Bathroom"
rent_fig["layout"]["xaxis12"]["title"] = "Bathroom"
rent_fig["layout"]["yaxis11"]["title"] = "Rent"
rent_fig["layout"]["yaxis12"]["title"] = "Size"

rent_fig["layout"]["xaxis13"]["title"] = "Point of Contact"
rent_fig["layout"]["xaxis14"]["title"] = "Point of Contact"
rent_fig["layout"]["yaxis13"]["title"] = "Rent"
rent_fig["layout"]["yaxis14"]["title"] = "Size"

rent_fig.update_layout(
    title=dict(text="Box Plots for bivariate analysis", font=dict(size=30)),
    height=1500,
)
rent_fig.show()


# Relationship between House Rent and House Size
x = rent_df["Rent"]
y = rent_df["Size"]
colors = rent_df["Size"]
sizes = rent_df["Size"]

plt.xlabel("Rent")
plt.ylabel("Size SqFt")
plt.ticklabel_format(style="plain")
plt.scatter(x, y, c=colors, s=sizes, alpha=0.3, cmap="RdBu")
plt.colorbar()
plt.show()

# Multivariate analysis
# Multivariate analysis of Size vs Rent and BHK
fig = make_subplots(
    rows=1, cols=1, subplot_titles=("Scatter Plot of Size vs Rent and BHK")
)
fig.append_trace(
    go.Scatter(
        x=rent_df["BHK"].values,
        y=rent_df["Size"].values,
        name="Size vs Rent and BHK",
        mode="markers",
        marker_color=rent_df["BHK"],
        marker_size=rent_df["Rent"] / 20000,
    ),
    row=1,
    col=1,
)
fig.update_layout(legend={"itemsizing": "trace"})
fig.update_layout(
    showlegend=False,
    title=dict(text="Scatter Plots for Multivariate Analysis", font=dict(size=30)),
    height=1500,
)
fig["layout"]["xaxis1"]["title"] = "BHK"
fig["layout"]["yaxis1"]["title"] = "Size in SqFt"
fig.show()

# Analysis using K-means clustering
rent_kmeans = rent_df.loc[:, ["Rent", "Size"]].values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(rent_kmeans)
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)

# plot graph of elbow method
sns.lineplot(wcss, marker="o", color="red")
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

# Optimal clustering is 3

# K-Means operations
kmeans = KMeans(n_clusters=3, init="k-means++", random_state=43)
SqFt_kmeans = kmeans.fit_predict(rent_kmeans)

# visualising the Kmeans
plt.scatter(rent_kmeans[:, 0], rent_kmeans[:, 1], s=10, c="mediumpurple")
plt.scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c="red"
)
plt.title("Clusters of Properties")
plt.xlabel("Rent")
plt.ylabel("Size in SqFt")
plt.show()
