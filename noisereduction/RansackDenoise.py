import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.signal import find_peaks

from pyobjtools import ObjFile

WORK_DIR = './DeNoiseWork/RANSACK'


class RansackDenoise:
    def __init__(self, obj_file):
        if not obj_file.lower().endswith(".obj"):
            raise TypeError("obj_file must end with .obj")

        if not os.path.isfile(obj_file):
            raise FileNotFoundError(f"The file '{obj_file}' does not exist.")

        self.file_name = os.path.splitext(os.path.basename(obj_file))[0]
        self.working_dir = os.path.join(WORK_DIR, self.file_name)
        os.makedirs(self.working_dir, exist_ok=True)

        self.obj = ObjFile(obj_file)
        self.obj.convert_to_csv(self.working_dir)

        self.vertices_path = os.path.join(self.working_dir, self.file_name + "/vertices.csv")
        print(self.vertices_path)

    def run_ransack(self):
        # Load vertex data
        vertices_filename = self.vertices_path

        df = pd.read_csv(vertices_filename, header=None, names=["VertexID", "VertexType", "X", "Y", "Z"])
        points = df[["X", "Y", "Z"]].to_numpy()

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Detect plane using RANSAC
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                 ransac_n=3,
                                                 num_iterations=1000)

        [a, b, c, d] = plane_model
        print(f"Detected plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
        print(f"Detected plane points: {len(inliers)} / {len(points)}")

        # Extract plane points (noise points)
        plane_points = df.iloc[inliers]

        # Save detected plane points to noise.csv
        os.makedirs(f"{self.working_dir}/noise", exist_ok=True)
        noise_path = f"{self.working_dir}/noise/noise.csv"
        plane_points.to_csv(noise_path, index=False, header=False)
        print(f"✅ Saved detected plane points to {self.working_dir}/noise/noise.csv")

        remove_set = {int(line[0]) for line in csv.reader(open(noise_path))}

        self.obj.remove_vertices_list_by_ids(remove_set)

        output_obj = f'{self.working_dir}/{self.file_name}_ransack_noise_reduced.obj'
        self.obj.write_obj_file(output_obj)
        print(f"Noise reduced OBJ saved as: {output_obj}")
        return output_obj

    def run_cuboid_detection(self, margin=0.02):
        """
        Detect a cuboid-like region around a plane (with thickness) and remove its points.
        margin: distance from plane within which points are included (thickness of cuboid).
        """
        # Load vertex data
        df = pd.read_csv(self.vertices_path, header=None, names=["VertexID", "VertexType", "X", "Y", "Z"])
        points = df[["X", "Y", "Z"]].to_numpy()

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Step 1: Fit a base plane (RANSAC, done only once)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000
        )

        a, b, c, d = plane_model
        print(f"Detected base plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

        # Step 2: Compute distance of every point from this plane
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(
            a ** 2 + b ** 2 + c ** 2)

        # Step 3: Select points within a cuboid margin (thickness)
        cuboid_mask = distances <= margin
        cuboid_points = df[cuboid_mask]

        print(f"Detected cuboid region points: {cuboid_points.shape[0]} / {points.shape[0]} (margin={margin})")

        # Step 4: Save cuboid (noise) points
        os.makedirs(f"{self.working_dir}/noise", exist_ok=True)
        noise_path = f"{self.working_dir}/noise/noise.csv"
        cuboid_points.to_csv(noise_path, index=False, header=False)
        print(f"✅ Saved cuboid (noise) points to {noise_path}")

        # Step 5: Remove those points from the OBJ model
        remove_set = {int(line[0]) for line in csv.reader(open(noise_path))}
        self.obj.remove_vertices_list_by_ids(remove_set)

        # Step 6: Save the cleaned OBJ
        output_obj = f'{self.working_dir}/{self.file_name}_cuboid_noise_reduced.obj'
        self.obj.write_obj_file(output_obj)
        print(f"Noise reduced OBJ saved as: {output_obj}")
        return output_obj

    def estimate_margin_from_plane(self, sample_size=500):
        """
        Detects the main plane using RANSAC and estimates a suitable cuboid margin
        based on nearby off-plane points.
        Returns (plane_model, margin)
        """

        # Load vertex data
        df = pd.read_csv(self.vertices_path, header=None, names=["VertexID", "VertexType", "X", "Y", "Z"])
        points = df[["X", "Y", "Z"]].to_numpy()

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Step 1: Fit a base plane using RANSAC
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000
        )

        a, b, c, d = plane_model
        print(f"Detected base plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

        # Step 2: Estimate adaptive margin from neighborhood
        plane_points = points[inliers]
        outlier_points = np.delete(points, inliers, axis=0)

        if len(outlier_points) == 0:
            print("⚠️ No off-plane points found; using default small margin = 0.01")
            return plane_model, 0.01

        # Build KD-tree for off-plane points
        outlier_pcd = o3d.geometry.PointCloud()
        outlier_pcd.points = o3d.utility.Vector3dVector(outlier_points)
        outlier_tree = o3d.geometry.KDTreeFlann(outlier_pcd)

        # Sample subset of plane points
        sample_size = min(sample_size, len(plane_points))
        sampled_points = plane_points[np.random.choice(len(plane_points), sample_size, replace=False)]

        distances = []
        for p in sampled_points:
            # Search for nearest neighbor (k=1)
            [_, _, dists] = outlier_tree.search_knn_vector_3d(p, 1)
            if dists:
                dist = np.sqrt(dists[0])
                # Discard unrealistic large distances (e.g., > 5 cm)
                if dist < 0.05:
                    distances.append(dist)

        if len(distances) == 0:
            print("⚠️ No valid neighbor distances found; using fallback margin = 0.01")
            return plane_model, 0.01

        # Remove extreme values (keep 90th percentile)
        filtered = np.array(distances)
        cutoff = np.percentile(filtered, 90)
        filtered = filtered[filtered <= cutoff]

        # Compute median and scale slightly
        margin = np.median(filtered) * 1.1  # add 10% buffer
        margin = np.clip(margin, 0.002, 0.02)  # clamp to safe range (2mm – 2cm)

        print(f"Estimated adaptive margin: {margin:.5f}")

        return margin

    def estimate_margin_from_histogram(self, normalize_result=False, show_cutoff_fig=False):

        df = pd.read_csv(self.vertices_path, header=None, names=["VertexID", "VertexType", "X", "Y", "Z"])
        points = df[['X', 'Y', 'Z']].to_numpy()

        # === Option 1: use absolute Y (if object is upside down) ===
        Y = np.abs(points[:, 1])

        # === Option 2 (optional): normalize Y for rotation-invariant scale ===
        # uncomment this if your models have very different Y ranges
        if normalize_result:
            Y_min, Y_max = np.min(Y), np.max(Y)
            Y = (Y - Y_min) / (Y_max - Y_min)

        # === Build histogram ===
        bins = 200
        hist, bin_edges = np.histogram(Y, bins=bins)

        # === Smooth histogram ===
        window = 5
        smoothed = np.convolve(hist, np.ones(window) / window, mode='same')

        # === Find peaks ===
        peaks, _ = find_peaks(smoothed, distance=10)

        # === Detect cutoff (trough between two biggest peaks) ===
        cutoff = None
        if len(peaks) >= 2:
            top_peaks = sorted(peaks, key=lambda p: smoothed[p], reverse=True)[:2]
            top_peaks.sort()
            left, right = top_peaks
            trough_idx = np.argmin(smoothed[left:right])
            cutoff = bin_edges[trough_idx]
        else:
            cutoff = np.percentile(Y, 20)

        print(f"Detected cutoff (margin) at |Y| = {cutoff:.3f}")

        # === Mark noise points ===
        is_noise = np.abs(points[:, 1]) < cutoff
        noise_points = points[is_noise]
        object_points = points[~is_noise]

        print(f"Noise points: {len(noise_points)}, Object points: {len(object_points)}")

        # === Plot histogram and cutoff ===
        plt.figure(figsize=(8, 5))
        plt.plot(bin_edges[:-1], smoothed, label="Smoothed histogram")
        plt.axvline(cutoff, color='r', linestyle='--', label=f"Cutoff = {cutoff:.3f}")
        plt.title("|Y|-axis histogram (detecting plate/noise region)")
        plt.xlabel("|Y| value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()

        plt.savefig(f'{self.working_dir}/noise_histogram.png')
        if show_cutoff_fig:
            plt.show()
        # # Optional: save separated points for visualization
        # np.savetxt("noise_points.csv", noise_points, delimiter=",", fmt="%.6f")
        # np.savetxt("object_points.csv", object_points, delimiter=",", fmt="%.6f")
        # print("Saved 'noise_points.csv' and 'object_points.csv'.")
        return cutoff


if __name__ == '__main__':
    noiseModel = RansackDenoise('./temp/HDBSCAN/texturedMesh/texturedMesh_noise_reduced.obj')
    print(noiseModel.run_ransack())
