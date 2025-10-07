import pandas as pd
import numpy as np
import os
import hdbscan

def cluster_points(csv_path, min_cluster_size=10, min_samples=None, output_dir=None):
    """
    Cluster 3D vertices using HDBSCAN and separate object vs noise points.

    Parameters:
        csv_path (str): Path to the input CSV file (no header, columns = vertexID, VertexType, X, Y, Z)
        min_cluster_size (int): Minimum size of clusters. HDBSCAN will consider smaller clusters as noise.
        min_samples (int, optional): HDBSCAN parameter to control how conservative the clustering is.
        output_dir (str, optional): Directory to save results. Defaults to same as input file.

    Output:
        Saves two CSVs:
          - object_points.csv
          - noise_points.csv
    """

    # Load CSV without header
    df = pd.read_csv(csv_path, header=None)
    df.columns = ['vertexID', 'VertexType', 'X', 'Y', 'Z']

    # Extract coordinates
    coords = df[['X', 'Y', 'Z']].values

    # Run HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(coords)
    df['Cluster'] = cluster_labels

    # Separate noise points (label == -1)
    noise_df = df[df['Cluster'] == -1]
    clusters_df = df[df['Cluster'] != -1]

    # Determine the main (largest) cluster → considered the "object"
    main_cluster_label = clusters_df['Cluster'].value_counts().idxmax() if not clusters_df.empty else None

    object_points = clusters_df[clusters_df['Cluster'] == main_cluster_label] if main_cluster_label is not None else pd.DataFrame()
    noise_points = df[~df.index.isin(object_points.index)]

    # Output directory
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    object_path = os.path.join(output_dir, "object_points.csv")
    noise_path = os.path.join(output_dir, "noise_points.csv")

    object_points.to_csv(object_path, index=False, header=False)
    noise_points.to_csv(noise_path, index=False, header=False)

    print(f"✅ Object points saved to: {object_path}")
    print(f"✅ Noise points saved to: {noise_path}")
    print(f"Clusters found: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")


if __name__ == "__main__":
    cluster_points(
        '../outputs/mug/vertices.csv',
        min_cluster_size=50,   # adjust depending on object size
        min_samples=None,
        output_dir='../outputs/mug/'
    )
