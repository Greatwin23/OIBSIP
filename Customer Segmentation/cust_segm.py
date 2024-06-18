import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

#Load dataset
data = pd.read_csv(r"C:\Users\great\ifood_df.csv")
cl = data.isna().any()
print(cl)
print(data.head(5))
print(data.shape)

#Calculate Descriptive Statistics
mean_total = data['MntTotal'].mean()
print("Average products total is",mean_total)
mean_reg = data['MntRegularProds'].mean()
print("Average Regular products total is",mean_reg)

# Calculate the frequency of each value for NumCatalogPurchases
frequency_Catalog = data['NumCatalogPurchases'].value_counts()

# Calculate the frequency of each value for NumStorePurchases
frequency_Store = data['NumStorePurchases'].value_counts()

# Calculate the frequency of each value for NumDealsPurchases
frequency_Deals = data['NumDealsPurchases'].value_counts()

# Calculate the frequency of each value for Recency
frequency_Recency = data['Recency'].value_counts()

#Range Recency
Range_Recency = (data['Recency'].min(), data['Recency'].max())
print("Frequency of NumCatalogPurchases:")
print(frequency_Catalog)

print("\nFrequency of NumStorePurchases:")
print(frequency_Store)

print("\nFrequency of NumDealsPurchases:")
print(frequency_Deals)

print("\nFrequency of Recency:")
print(frequency_Recency)
print("Range of Recency",Range_Recency )

# Select relevant features for clustering
features = [
    'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 
    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
    'NumDealsPurchases', 'NumWebVisitsMonth'
]

# Preprocess the data: Handle missing values
data = data[features].dropna()

# Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Choose the optimal number of clusters (e.g., 4)
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Analyze the clusters
cluster_summary = data.groupby('Cluster').mean()
print(cluster_summary)

# Visualize the clusters
plt.figure(figsize=(15, 10))
spending_categories = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
for i, category in enumerate(spending_categories, start=1):
    plt.subplot(2, 3, i)
    sns.scatterplot(data=data, x='Income', y=category, hue='Cluster', palette='viridis')
    plt.title(f'Customer Segments based on Income and {category}')
    plt.xlabel('Income')
    plt.ylabel(category)
    plt.tight_layout()

plt.show()

# Visualize the clusters for categorical features
plt.figure(figsize=(15, 8))
categorical_features = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumDealsPurchases', 'NumWebVisitsMonth']
for i, feature in enumerate(categorical_features, start=1):
    plt.subplot(2, 3, i)
    sns.barplot(x='Cluster', y=feature, data=data, hue='Cluster', palette='viridis', dodge=False)
    plt.title(f'Average {feature} by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(f'Average {feature}')
    plt.legend().remove()  # Remove the legend since hue is just for coloring
    plt.tight_layout()

plt.show()


# Save the clustered data to a new CSV file
output_path = r"C:\Users\Public\Documents/ifood_clustered.csv"
data.to_csv(output_path, index=False)

# Assuming 'data' is your DataFrame with clustering results
# Analyze the clusters
cluster_summary = data.groupby('Cluster').mean()
print(cluster_summary)
