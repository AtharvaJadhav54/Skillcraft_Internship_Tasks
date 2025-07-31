# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

# --- 0. Verify File Existence ---
# List files in the current directory to ensure the dataset is available
print("Files in the current directory:")
print(os.listdir('.'))


# --- 1. Load and Prepare the Data ---
try:
    df = pd.read_csv('Mall_Customers.csv')
    print("\nDataset loaded successfully.")
except FileNotFoundError:
    print("\nError: Mall_Customers.csv not found.")
    # Exit the script if the file is not found
    exit()

# Rename columns for easier access
df.rename(columns={
    'Annual Income (k$)': 'Annual_Income',
    'Spending Score (1-100)': 'Spending_Score'
}, inplace=True)

print("\nFirst 5 rows of the dataset:")
print(df.head())

# Select the features for clustering
features = ['Annual_Income', 'Spending_Score']
X = df[features]

# Standardize the features
# This ensures that all features are on the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nFeatures have been scaled for clustering.")


# --- 2. Find the Optimal Number of Clusters (k) ---
print("\nDetermining the optimal number of clusters...")
wcss = []  # Within-Cluster Sum of Squares
silhouette_scores = []
k_range = range(2, 11) # Start from 2 for silhouette score

for k in k_range:
    # K-means model
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # Store metrics
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plotting the Elbow Method
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)

# Plotting Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o', linestyle='--', color='r')
plt.title('Silhouette Scores')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)

plt.tight_layout()
plt.savefig('optimal_k_plots.png')
print("Plots for determining optimal k saved as 'optimal_k_plots.png'.")

# The "elbow" in the WCSS plot and the peak in the silhouette score plot suggest the optimal k.
# For this dataset, it is typically 5.
optimal_k = 5
print(f"\nSelected optimal number of clusters: {optimal_k}")


# --- 3. Apply K-Means and Visualize Segments ---
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nK-Means clustering complete. Cluster labels added to the DataFrame.")

# Get cluster centroids and scale them back to the original data scale
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# Create a scatter plot of the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df,
    x='Annual_Income',
    y='Spending_Score',
    hue='Cluster',
    palette='viridis',
    s=100,
    alpha=0.8,
    legend='full'
)

# Plot the centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=300,
    c='red',
    marker='X',
    label='Centroids'
)

plt.title('Customer Segments based on Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Customer Group')
plt.grid(True)
plt.savefig('customer_segments_plot.png')
print("Customer segments visualization saved as 'customer_segments_plot.png'.")


# --- 4. Analyze and Interpret the Clusters ---
cluster_summary = df.groupby('Cluster')[['Age', 'Annual_Income', 'Spending_Score']].mean()
print("\n--- Cluster Profiles (Mean Values) ---")
print(cluster_summary)

# Define descriptive names for the clusters based on their characteristics
cluster_names = {
    0: 'High Income, Low Spending',
    1: 'Average',
    2: 'High Income, High Spending (Target)',
    3: 'Low Income, Low Spending',
    4: 'Low Income, High Spending'
}
df['Segment_Name'] = df['Cluster'].map(cluster_names)

print("\n--- Cluster Interpretation ---")
for i in sorted(cluster_names.keys()):
    print(f"Cluster {i}: {cluster_names[i]}")

# Save the final DataFrame with cluster labels and names to a CSV file
df.to_csv('customer_segments_output.csv', index=False)
print("\nFinal segmented data saved to 'customer_segments_output.csv'.")
print("\nAnalysis complete! 🎉")