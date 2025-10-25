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
        import numpy as np
        import pandas as pd
        import open3d as o3d
        import os

        # Load vertex data
        df = pd.read_csv(self.vertices_path, header=None, names=["VertexID", "VertexType", "X", "Y", "Z"])
        points = df[["X", "Y", "Z"]].to_numpy()

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        planes = []
        remaining = pcd

        # Run RANSAC multiple times to detect multiple planes
        for _ in range(5):  # You can increase if there are more layers
            if len(remaining.points) < 100:
                break

            plane_model, inliers = remaining.segment_plane(
                distance_threshold=0.01,
                ransac_n=3,
                num_iterations=1000
            )

            if len(inliers) < 100:
                break  # too small, stop

            plane_points = np.asarray(remaining.points)[inliers]
            mean_y = plane_points[:, 1].mean()

            planes.append({
                "model": plane_model,
                "inliers": inliers,
                "mean_y": mean_y,
                "points": plane_points
            })

            # remove current plane to find others
            remaining = remaining.select_by_index(inliers, invert=True)

        if not planes:
            print("❌ No planes detected.")
            return None

        # Find the lowest plane (smallest mean Y)
        lowest_plane = min(planes, key=lambda p: p["mean_y"])
        a, b, c, d = lowest_plane["model"]
        print(f"✅ Lowest plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
        print(f"Plane mean Y: {lowest_plane['mean_y']:.4f}")
        print(f"Detected plane points: {len(lowest_plane['inliers'])} / {len(points)}")

        # Save detected lowest plane points as noise
        plane_points_df = pd.DataFrame(lowest_plane["points"], columns=["X", "Y", "Z"])
        os.makedirs(f"{self.working_dir}/noise", exist_ok=True)
        noise_path = f"{self.working_dir}/noise/noise.csv"
        plane_points_df.to_csv(noise_path, index=False, header=False)
        print(f"✅ Saved lowest plane points to {noise_path}")

        # Get vertex IDs corresponding to the removed points
        noise_ids = df[df[["X", "Y", "Z"]].apply(tuple, axis=1).isin(
            [tuple(p) for p in lowest_plane["points"]]
        )]["VertexID"]

        remove_set = set(noise_ids.astype(int))
        self.obj.remove_vertices_list_by_ids(remove_set)

        # Write output OBJ
        output_obj = f'{self.working_dir}/{self.file_name}_ransack_lowest_plate_removed.obj'
        self.obj.write_obj_file(output_obj)
        print(f"✅ Noise reduced OBJ saved as: {output_obj}")

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

    def detect_noise_enclosed_shape(
            self,
            bin_size=0.005,
            quantile_cut=0.25,
            high_quantile=0.9,  # stricter high variance cutoff
            min_region_size=5,  # ignore tiny high-variance blobs (in grid cells)
            visualize=True):
        """
        Detect enclosed high-variance regions in X-Z (top view), keep only those
        regions (boundary + enclosed interior), export noise points and remove them
        from the OBJ.

        IMPORTANT: This function assumes the CSV read by `self.vertices_path`
        includes a VertexID column that matches the vertex IDs expected by your
        `self.obj.remove_vertices_list_by_ids()` method.
        """

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from collections import deque

        # === Load points ===
        df = pd.read_csv(self.vertices_path, header=None, names=["VertexID", "VertexType", "X", "Y", "Z"])
        X = df["X"].to_numpy()
        Y = df["Y"].to_numpy()
        Z = df["Z"].to_numpy()

        # === Bin X/Z into integer grid coordinates ===
        x_bins = np.floor(X / bin_size).astype(int)
        z_bins = np.floor(Z / bin_size).astype(int)
        bin_keys = np.stack((x_bins, z_bins), axis=1)
        unique_bins, inv_idx = np.unique(bin_keys, axis=0, return_inverse=True)

        # === Y-range (thickness) per unique (x,z) bin ===
        n_bins = len(unique_bins)
        y_range = np.zeros(n_bins, dtype=float)
        for i in range(n_bins):
            ys = Y[inv_idx == i]
            if len(ys) > 1:
                y_range[i] = ys.max() - ys.min()
            else:
                y_range[i] = 0.0

        # === Build dense grid covering all occupied bins ===
        x_min, z_min = unique_bins.min(axis=0)
        x_max, z_max = unique_bins.max(axis=0)
        gx = x_max - x_min + 1
        gz = z_max - z_min + 1
        grid_y_range = np.zeros((gx, gz), dtype=float)

        # Fill grid with y_range values (0 where no points)
        for idx, (bx, bz) in enumerate(unique_bins):
            ix, iz = bx - x_min, bz - z_min
            grid_y_range[ix, iz] = y_range[idx]

        # === Smooth Y-range map (3x3 mean) to reduce spiky noise ===
        smoothed = grid_y_range.copy()
        # Only iterate internal cells to avoid boundary index checks
        for i in range(1, gx - 1):
            for j in range(1, gz - 1):
                smoothed[i, j] = grid_y_range[i - 1:i + 2, j - 1:j + 2].mean()

        # === Thresholds ===
        nonzero_vals = smoothed[smoothed > 0]
        if nonzero_vals.size == 0:
            print("No occupied bins found. Nothing to remove.")
            return None

        range_cutoff = np.quantile(nonzero_vals, high_quantile)
        y_cutoff = np.quantile(Y, quantile_cut)  # kept for info but not used for keeping points
        print(f"Y-range cutoff (Q{high_quantile * 100:.0f}): {range_cutoff:.6f}")
        print(f"Y quantile cutoff (Q{quantile_cut * 100:.0f}): {y_cutoff:.4f}")

        # === Initial high-variance mask (candidate boundary) ===
        grid_high = smoothed >= range_cutoff

        # === Remove tiny isolated high-variance blobs (morphological cleaning) ===
        visited = np.zeros_like(grid_high, dtype=bool)
        clean_high = np.zeros_like(grid_high, dtype=bool)

        def neighbors8(i, j):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < gx and 0 <= nj < gz:
                        yield ni, nj

        for i in range(gx):
            for j in range(gz):
                if grid_high[i, j] and not visited[i, j]:
                    q = deque([(i, j)])
                    visited[i, j] = True
                    comp = []
                    while q:
                        ci, cj = q.popleft()
                        comp.append((ci, cj))
                        for ni, nj in neighbors8(ci, cj):
                            if grid_high[ni, nj] and not visited[ni, nj]:
                                visited[ni, nj] = True
                                q.append((ni, nj))
                    if len(comp) >= min_region_size:
                        for (ci, cj) in comp:
                            clean_high[ci, cj] = True

        grid_high = clean_high

        # === Flood-fill connected low-variance components and mark enclosed ones ===
        filled_mask = np.zeros_like(grid_high, dtype=bool)
        visited = np.zeros_like(grid_high, dtype=bool)

        for i in range(gx):
            for j in range(gz):
                if not grid_high[i, j] and not visited[i, j]:
                    q = deque([(i, j)])
                    visited[i, j] = True
                    comp = []
                    touches_border = False
                    while q:
                        ci, cj = q.popleft()
                        comp.append((ci, cj))
                        if ci == 0 or ci == gx - 1 or cj == 0 or cj == gz - 1:
                            touches_border = True
                        for ni, nj in neighbors8(ci, cj):
                            if not grid_high[ni, nj] and not visited[ni, nj]:
                                visited[ni, nj] = True
                                q.append((ni, nj))
                    # If component does NOT touch grid border, it is enclosed -> keep it
                    if not touches_border:
                        for (ci, cj) in comp:
                            filled_mask[ci, cj] = True

        # === Final keep mask at grid level: keep boundary OR enclosed interior ===
        keep_mask = grid_high | filled_mask

        # === Map keep mask back to each unique bin, then to each point ===
        keep_per_unique_bin = np.zeros(n_bins, dtype=bool)
        for idx, (bx, bz) in enumerate(unique_bins):
            ix, iz = bx - x_min, bz - z_min
            keep_per_unique_bin[idx] = keep_mask[ix, iz]

        # Boolean mask for each point
        is_enclosed_or_high = keep_per_unique_bin[inv_idx]
        keep_points = is_enclosed_or_high  # <<--- CRITICAL: only keep these
        noise_points = ~keep_points

        # Logging counts
        n_noise = int(noise_points.sum())
        n_keep = int(keep_points.sum())
        print(f"Detected {n_noise} noise points and {n_keep} kept (object) points")

        # === Visualization (top-view heatmap + overlays) ===
        if visualize:
            plt.figure(figsize=(9, 6))
            plt.imshow(smoothed.T, origin='lower', cmap='viridis')
            plt.title("Smoothed Y-range heatmap (top view)")
            plt.colorbar(label="Y-range (thickness)")

            hi = np.argwhere(grid_high)
            if hi.size:
                plt.scatter(hi[:, 0], hi[:, 1], color='red', s=3, label="High variance (boundary)")

            fi = np.argwhere(filled_mask)
            if fi.size:
                plt.scatter(fi[:, 0], fi[:, 1], color='blue', s=3, label="Enclosed region (kept)")

            # Optionally show which unique bins are actually occupied (light gray)
            occ = np.argwhere(grid_y_range > 0)
            if occ.size:
                plt.scatter(occ[:, 0], occ[:, 1], color='lightgray', s=1, alpha=0.6, label="Occupied bins")

            plt.legend(loc='upper right')
            plt.xlabel("X bin")
            plt.ylabel("Z bin")
            plt.tight_layout()
            plt.show()

        # === Export noise points (use VertexID reliably) ===
        # Select original columns and rows for noise points
        noise_df = df.loc[noise_points, ["VertexID", "VertexType", "X", "Y", "Z"]].copy()

        # Make sure VertexID is integer-like before removing
        try:
            noise_df["VertexID"] = noise_df["VertexID"].astype(int)
        except Exception:
            # If conversion fails, fall back to using DataFrame index (but prefer VertexID)
            noise_df["VertexID"] = noise_df.index.astype(int)
            print("Warning: VertexID column not integer-convertible; using DataFrame index for removal.")

        noise_path = f"{self.working_dir}/noise_points.csv"
        noise_df.to_csv(noise_path, index=False, header=False)
        print(f"Saved noise points to {noise_path} (rows: {len(noise_df)})")

        # Remove by VertexID values (as integers)
        remove_set = set(noise_df["VertexID"].tolist())
        # Call your OBJ removal function
        self.obj.remove_vertices_list_by_ids(remove_set)

        # Write cleaned OBJ file and return path
        cleaned_path = f"{self.working_dir}/{self.file_name}_noise_enclosed.obj"
        self.obj.write_obj_file(cleaned_path)
        print(f"Wrote cleaned OBJ to: {cleaned_path}")

        return cleaned_path


    def detect_noise_by_ransac_plane(self,
                                     max_iters=1500,
                                     distance_thresh=0.003,  # distance from plane to consider inlier
                                     min_inlier_frac=0.02,  # require at least this fraction of points to accept plane
                                     sample_region_frac=0.25  # sample only lowest fraction of points to focus on plate
                                     ):
        import csv, random
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        # load
        df = pd.read_csv(self.vertices_path, header=None, names=["VertexID", "VertexType", "X", "Y", "Z"])
        points = df[['VertexID', 'VertexType', 'X', 'Y', 'Z']].to_numpy()
        X = points[:, 2];
        Y = points[:, 3];
        Z = points[:, 4]

        # focus sampling on lowest part of cloud (likely contains plate)
        y_cut = np.quantile(Y, sample_region_frac)
        sample_mask = Y <= y_cut
        sample_idx = np.where(sample_mask)[0]
        if len(sample_idx) < 3:
            sample_idx = np.arange(len(points))

        best_plane = None
        best_inliers = None
        best_count = 0

        pts_xyz = np.stack([X, Y, Z], axis=1)

        for it in range(max_iters):
            # random 3 points (avoid collinearity)
            ids = np.random.choice(sample_idx, size=3, replace=False)
            p0, p1, p2 = pts_xyz[ids]
            v1 = p1 - p0
            v2 = p2 - p0
            n = np.cross(v1, v2)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-9:
                continue
            n = n / n_norm
            d = -np.dot(n, p0)
            # distances of all points to plane
            dist = np.abs(np.dot(pts_xyz, n) + d)
            inliers = np.where(dist <= distance_thresh)[0]
            cnt = len(inliers)
            if cnt > best_count:
                best_count = cnt
                best_inliers = inliers
                best_plane = (n.copy(), float(d))

        if best_inliers is None or best_count < max(3, int(min_inlier_frac * len(points))):
            print("RANSAC failed to find a dominant plane — no removal performed.")
            return None

        # choose plate inliers that are at the bottom (low Y)
        plate_inliers = best_inliers[Y[best_inliers] <= np.quantile(Y[best_inliers], 0.9)]

        print(f"RANSAC plane found with {len(best_inliers)} inliers; candidate plate inliers: {len(plate_inliers)}")

        # classify noise = plate_inliers (below plane within threshold and low Y)
        noise_mask = np.zeros(len(points), dtype=bool)
        noise_mask[plate_inliers] = True

        noise_points = points[noise_mask]
        object_points = points[~noise_mask]

        # save noise csv
        noise_path = f"{self.working_dir}/noise_points.csv"
        pd.DataFrame(noise_points, columns=["VertexID", "VertexType", "X", "Y", "Z"]).to_csv(noise_path, index=False,
                                                                                             header=False)
        print("Saved:", noise_path)

        # remove from OBJ
        remove_set = {int(line[0]) for line in csv.reader(open(noise_path))}
        self.obj.remove_vertices_list_by_ids(remove_set)
        cleaned_path = f"{self.working_dir}/{self.file_name}_noise_ransac.obj"
        self.obj.write_obj_file(cleaned_path)

        # visualization
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.scatter(points[:, 2], points[:, 3], s=3, c='lightgray', label='all (X vs Y)')
        ax.scatter(noise_points[:, 2], noise_points[:, 3], s=6, c='red', label='noise (plate)')
        ax.set_xlabel('X');
        ax.set_ylabel('Y');
        ax.legend();
        plt.title('RANSAC plate detection (X vs Y)')
        plt.show()

        return cleaned_path


if __name__ == '__main__':
    noiseModel = RansackDenoise('./temp/HDBSCAN/texturedMesh/texturedMesh_noise_reduced.obj')
    print(noiseModel.run_ransack())
