import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.signal import find_peaks
from collections import deque

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

    def detect_noise_by_y_range(self,
                                bin_size=0.005,  # how tightly X/Z bins are grouped
                                quantile_cut=0.25  # defines "object starts here" (Q1 usually)
                                ):
        import csv
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        # === Load points ===
        df = pd.read_csv(self.vertices_path, header=None, names=["VertexID", "VertexType", "X", "Y", "Z"])
        points = df[['VertexID', 'VertexType', 'X', 'Y', 'Z']].to_numpy()

        X = points[:, 2]
        Y = points[:, 3]
        Z = points[:, 4]

        # === Bin X and Z ===
        x_bins = np.floor(X / bin_size).astype(int)
        z_bins = np.floor(Z / bin_size).astype(int)

        # === Compute Y range per (X,Z) bin ===
        bin_keys = np.stack((x_bins, z_bins), axis=1)
        unique_bins, inv_idx = np.unique(bin_keys, axis=0, return_inverse=True)

        y_range_per_bin = np.zeros(len(unique_bins))
        for i in range(len(unique_bins)):
            ys = Y[inv_idx == i]
            y_range_per_bin[i] = ys.max() - ys.min() if len(ys) > 1 else 0.0

        # Map Y range to each point
        point_y_range = y_range_per_bin[inv_idx]

        # === Compute Y quantile cutoff ===
        y_cutoff = np.quantile(Y, quantile_cut)
        print(f"Y quantile cutoff (Q{quantile_cut * 100:.0f}): {y_cutoff:.4f}")

        # === Compute adaptive range threshold ===
        # Let’s use a data-driven cutoff for “large” variance
        # e.g. 75th percentile of all y-ranges
        range_cutoff = np.quantile(y_range_per_bin, 0.75)
        print(f"Adaptive Y-range cutoff (Q75): {range_cutoff:.6f}")

        # === Classification ===
        # Accept if: (1) has big range OR (2) Y above Q1
        is_object = (point_y_range >= range_cutoff) | (Y >= y_cutoff)
        is_noise = ~is_object

        noise_points = points[is_noise]
        object_points = points[is_object]

        print(f"Detected {len(noise_points)} noise points and {len(object_points)} object points")

        # === Save noise points ===
        noise_path = f"{self.working_dir}/noise_points.csv"
        pd.DataFrame(noise_points, columns=["VertexID", "VertexType", "X", "Y", "Z"]).to_csv(
            noise_path, index=False, header=False
        )

        print("Saved: noise_points.csv")

        # === Remove noise from OBJ ===
        remove_set = {int(line[0]) for line in csv.reader(open(noise_path))}
        self.obj.remove_vertices_list_by_ids(remove_set)
        cleaned_path = f'{self.working_dir}/{self.file_name}_noise_by_y_range.obj'
        self.obj.write_obj_file(cleaned_path)

        # === Plot heatmap of local Y-range ===
        plt.figure(figsize=(8, 6))
        plt.scatter(unique_bins[:, 0] * bin_size,
                    unique_bins[:, 1] * bin_size,
                    c=y_range_per_bin,
                    cmap='viridis',
                    s=8)
        plt.colorbar(label="Y range (vertical thickness)")
        plt.title("Local Y range across X–Z (high = object, low = plate noise)")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.tight_layout()
        plt.show()

        return cleaned_path

    def get_highest_variance_y_range(self, bin_size=0.005, use_variance=False):
        """
        Finds the (X,Z) region with the highest Y range (or variance) and returns its min & max Y.

        Parameters:
            vertices_path (str): path to the CSV file
            bin_size (float): controls grouping tightness
            use_variance (bool): if True, use Y variance instead of range

        Returns:
            (min_y, max_y, x_center, z_center)
        """
        # === Load points ===
        df = pd.read_csv(self.vertices_path, header=None, names=["VertexID", "VertexType", "X", "Y", "Z"])
        X = df["X"].to_numpy()
        Y = df["Y"].to_numpy()
        Z = df["Z"].to_numpy()

        # === Bin X and Z ===
        x_bins = np.floor(X / bin_size).astype(int)
        z_bins = np.floor(Z / bin_size).astype(int)
        bin_keys = np.stack((x_bins, z_bins), axis=1)
        unique_bins, inv_idx = np.unique(bin_keys, axis=0, return_inverse=True)

        # === Compute Y range/variance per bin ===
        metric_per_bin = np.zeros(len(unique_bins))
        for i in range(len(unique_bins)):
            ys = Y[inv_idx == i]
            if len(ys) > 1:
                metric_per_bin[i] = np.var(ys) if use_variance else ys.max() - ys.min()
            else:
                metric_per_bin[i] = 0.0

        # === Find bin with highest Y variance/range ===
        max_idx = np.argmax(metric_per_bin)
        ys_in_max_bin = Y[inv_idx == max_idx]

        min_y, max_y = ys_in_max_bin.min(), ys_in_max_bin.max()
        x_center = (unique_bins[max_idx, 0] + 0.5) * bin_size
        z_center = (unique_bins[max_idx, 1] + 0.5) * bin_size

        print(f"Highest variance/range bin at (X,Z)=({x_center:.3f},{z_center:.3f})")
        print(f"Y range: {min_y:.4f} → {max_y:.4f} (Δ={max_y - min_y:.4f})")

        return min_y, max_y, x_center, z_center

    def remove_points_below_min_y(self):
        min_y, max_y, x_center, z_center = self.get_highest_variance_y_range()
        # === Load vertices ===
        df = pd.read_csv(self.vertices_path, header=None, names=["VertexID", "VertexType", "X", "Y", "Z"])
        points = df[["VertexID", "VertexType", "X", "Y", "Z"]].to_numpy()

        Y = points[:, 3]

        # === Classify ===
        is_noise = Y < min_y
        is_object = ~is_noise

        noise_points = points[is_noise]
        object_points = points[is_object]

        print(f"Removed {len(noise_points)} points below min_y = {min_y:.4f}")
        print(f"Remaining object points: {len(object_points)}")

        # === Save noise points ===
        noise_path = f"{self.working_dir}/noise_points_below_y.csv"
        pd.DataFrame(noise_points, columns=["VertexID", "VertexType", "X", "Y", "Z"]).to_csv(
            noise_path, index=False, header=False
        )
        print(f"Saved: noise_points_below_y.csv")

        # === Remove vertices from OBJ ===
        remove_set = {int(line[0]) for line in csv.reader(open(noise_path))}
        self.obj.remove_vertices_list_by_ids(remove_set)
        cleaned_path = f'{self.working_dir}/{self.file_name}_above_min_y.obj'
        self.obj.write_obj_file(cleaned_path)

        # === Plot Y distribution ===
        plt.figure(figsize=(8, 5))
        plt.hist(Y, bins=60, alpha=0.6, color='gray', label='All points')
        plt.axvline(min_y, color='red', linestyle='--', label=f"min_y cutoff = {min_y:.3f}")
        plt.hist(object_points[:, 3], bins=60, alpha=0.7, color='green', label='Kept points')
        plt.xlabel("Y value")
        plt.ylabel("Count")
        plt.title("Y distribution before and after removal")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return cleaned_path


    def detect_noise_enclosed_shape(self,
            bin_size=0.005,
            quantile_cut=0.25,
            visualize=True):
        """
        Detects noise by identifying closed high-variance regions in Y
        (thickness) over the X-Z plane and keeping points inside them.

        Keeps:
            - Points within high-variance regions (object boundaries)
            - Points enclosed by those boundaries (interior)
            - Points above certain Y quantile (e.g. tall columns)

        Removes:
            - Low-variance regions connected to the outside (noise)
        """

        # === Load points ===
        df = pd.read_csv(self.vertices_path, header=None, names=["VertexID", "VertexType", "X", "Y", "Z"])
        X, Y, Z = df["X"].to_numpy(), df["Y"].to_numpy(), df["Z"].to_numpy()

        # === Bin X/Z ===
        x_bins = np.floor(X / bin_size).astype(int)
        z_bins = np.floor(Z / bin_size).astype(int)
        bin_keys = np.stack((x_bins, z_bins), axis=1)
        unique_bins, inv_idx = np.unique(bin_keys, axis=0, return_inverse=True)

        # === Compute Y range (thickness) for each (X,Z) bin ===
        y_range = np.zeros(len(unique_bins))
        for i in range(len(unique_bins)):
            ys = Y[inv_idx == i]
            y_range[i] = ys.max() - ys.min() if len(ys) > 1 else 0.0

        # === Adaptive thresholds ===
        range_cutoff = np.quantile(y_range, 0.75)
        y_cutoff = np.quantile(Y, quantile_cut)

        print(f"Y-range cutoff (Q75): {range_cutoff:.6f}")
        print(f"Y quantile cutoff (Q{quantile_cut * 100:.0f}): {y_cutoff:.4f}")

        # === Create 2D grid of bins ===
        x_min, z_min = unique_bins.min(axis=0)
        x_max, z_max = unique_bins.max(axis=0)
        gx, gz = x_max - x_min + 1, z_max - z_min + 1

        grid_y_range = np.zeros((gx, gz))
        grid_high = np.zeros((gx, gz), dtype=bool)
        for i, (bx, bz) in enumerate(unique_bins):
            ix, iz = bx - x_min, bz - z_min
            grid_y_range[ix, iz] = y_range[i]
            grid_high[ix, iz] = y_range[i] >= range_cutoff

        # === Fill enclosed regions (8-connectivity flood fill) ===
        filled_mask = np.zeros_like(grid_high, dtype=bool)
        visited = np.zeros_like(grid_high, dtype=bool)
        gx_range, gz_range = range(gx), range(gz)

        def neighbors8(i, j):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < gx and 0 <= nj < gz:
                        yield ni, nj

        label = 0
        for i in gx_range:
            for j in gz_range:
                if not grid_high[i, j] and not visited[i, j]:
                    label += 1
                    q = deque([(i, j)])
                    visited[i, j] = True
                    component = []
                    touches_border = False
                    while q:
                        ci, cj = q.popleft()
                        component.append((ci, cj))
                        if ci == 0 or ci == gx - 1 or cj == 0 or cj == gz - 1:
                            touches_border = True
                        for ni, nj in neighbors8(ci, cj):
                            if not grid_high[ni, nj] and not visited[ni, nj]:
                                visited[ni, nj] = True
                                q.append((ni, nj))
                    # Fill if enclosed (doesn't touch border)
                    if not touches_border:
                        for ci, cj in component:
                            filled_mask[ci, cj] = True

        # === Final keep mask: high variance OR enclosed ===
        keep_mask = grid_high | filled_mask

        # === Map bins back to points ===
        keep_per_bin = np.zeros(len(unique_bins), dtype=bool)
        for i, (bx, bz) in enumerate(unique_bins):
            ix, iz = bx - x_min, bz - z_min
            keep_per_bin[i] = keep_mask[ix, iz]

        # === Keep points that are: enclosed OR high variance OR tall ===
        is_enclosed_or_high = keep_per_bin[inv_idx]
        is_tall = Y >= y_cutoff
        keep_points = is_enclosed_or_high | is_tall
        noise_points = ~keep_points

        print(f"Detected {noise_points.sum()} noise points and {keep_points.sum()} object points")

        # === Visualization ===
        if visualize:
            plt.figure(figsize=(8, 6))
            plt.imshow(grid_y_range.T, origin='lower', cmap='viridis')
            plt.title("Y-range heatmap (top view)")
            plt.colorbar(label="Y-range (thickness)")

            # overlay high variance (boundary)
            hi = np.argwhere(grid_high)
            plt.scatter(hi[:, 0], hi[:, 1], color='red', s=3, label="High variance (boundary)")

            # overlay filled enclosed regions
            fi = np.argwhere(filled_mask)
            plt.scatter(fi[:, 0], fi[:, 1], color='white', s=3, label="Enclosed region (kept)")

            plt.legend()
            plt.xlabel("X bin")
            plt.ylabel("Z bin")
            plt.tight_layout()
            plt.show()

        return keep_points


if __name__ == '__main__':
    noiseModel = RansackDenoise('./temp/HDBSCAN/texturedMesh/texturedMesh_noise_reduced.obj')
    print(noiseModel.run_ransack())
