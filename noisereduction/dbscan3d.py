import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import os

def cluster_points(csv_path, eps=0.5, min_samples=5, output_dir=None):
    """
    Cluster 3D vertices using DBSCAN and separate object vs noise points.

    Parameters:
        csv_path (str): Path to the input CSV file (no header, columns = vertexID, VertexType, X, Y, Z)
        eps (float): Max distance between points to be considered neighbors.
        min_samples (int): Minimum number of points to form a dense region.
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

    # Run DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    df['Cluster'] = db.labels_

    # Separate noise points (label == -1)
    noise_df = df[df['Cluster'] == -1]
    clusters_df = df[df['Cluster'] != -1]

    # Determine the main (largest) cluster → considered the "object"
    main_cluster_label = (
        clusters_df['Cluster'].value_counts().idxmax()
        if not clusters_df.empty else None
    )

    object_points = clusters_df[clusters_df['Cluster'] == main_cluster_label] if main_cluster_label is not None else pd.DataFrame()
    noise_points = df[~df.index.isin(object_points.index)]

    # Output directory
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save the results
    object_path = os.path.join(output_dir, "object_points.csv")
    noise_path = os.path.join(output_dir, "noise_points.csv")

    object_points.to_csv(object_path, index=False, header=False)
    noise_points.to_csv(noise_path, index=False, header=False)

    print(f"✅ Object points saved to: {object_path}")
    print(f"✅ Noise points saved to: {noise_path}")
    print(f"Clusters found: {len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)}")

if __name__ == "__main__":
   cluster_points('../outputs/cube2/vertices.csv',eps=1.3,min_samples=3,output_dir='../outputs/cube2/')