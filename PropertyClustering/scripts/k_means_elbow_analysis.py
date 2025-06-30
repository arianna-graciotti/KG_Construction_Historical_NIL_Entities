import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

X = pd.read_csv("/data/properties/human/1_datasets/human_properties_with_labels_no_p31.csv")
data = X[["count", ]]


sse = {}
for k in range(1, 100):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
    data["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

# Create figure with proper size
plt.figure(figsize=(16, 10))

# Plot the SSE vs number of clusters
plt.plot(list(sse.keys()), list(sse.values()), 'o-', linewidth=2, markersize=6, color='#3498db')

# Customize x-axis with detailed labels
plt.xticks(np.arange(0, 101, 5))  # Major ticks every 5 units
plt.grid(True, alpha=0.3)

# Add minor ticks for every value of k
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(1))
plt.grid(True, which='minor', alpha=0.1)  # Light grid for minor ticks

# Add vertical lines at each k in the critical region
for k in range(2, 15):
    plt.axvline(x=k, color='gray', linestyle='--', alpha=0.2)

# Enhance axes and title
plt.xlabel("Number of clusters (k)", fontsize=14, fontweight='bold')
plt.ylabel("Sum of Squared Errors (SSE)", fontsize=14, fontweight='bold')
plt.title("Elbow Method for Optimal k Selection", fontsize=16, fontweight='bold')

# Save high-resolution image
plt.savefig("/home/arianna/PycharmProjects/NIL_Grounding/data/properties/human/elbow_plot_detailed.png", 
           dpi=300, bbox_inches='tight')

# Display the plot
plt.show()