# import all needed packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# import the rental dataset using pandas
rent_df = pd.read_csv("/Users/izzyabate/Downloads/archive (1)/House_Rent_Dataset.csv")

# check the dataset for null and duplicated values
rent_df.isnull().sum()
rent_df.duplicated().sum()

# Exploring the dataset
print(rent_df.head())
print(rent_df.describe())
print(rent_df.columns)

# Visualize the correlation your data and identify variables for further analysis
rent_pairs = sns.PairGrid(rent_df)
rent_pairs.map(sns.scatterplot)

# Plot the number of properties in each city available for rent
sns.set_context("poster", font_scale=0.6)
plt.figure(figsize=(10, 6))
plt.xlabel("City")
plt.ylabel("Number of Properties")
ax = rent_df["City"].value_counts().plot(kind="bar", color="hotpink", rot=0)

for p in ax.patches:
    ax.annotate(
        int(p.get_height()),
        (p.get_x() + 0.25, p.get_height() - 100),
        ha="center",
        va="bottom",
        color="white",
    )
plt.show()


# Scatter plot showing the Rent versus the Size in SqFt
plt.scatter(rent_df["Size"], rent_df["Rent"], s=2, c="crimson")
plt.xlabel("Size SqFt")
plt.ylabel("Rent")
plt.show()

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
