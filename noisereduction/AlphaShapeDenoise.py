import os

import numpy as np
import open3d as o3d

WORK_DIR = './DeNoiseWork/AlphaShapeDenoise'


class AlphaShapeDenoise:
    def __init__(self, obj_file):
        if not obj_file.lower().endswith(".obj"):
            raise TypeError("obj_file must end with .obj")

        if not os.path.isfile(obj_file):
            raise FileNotFoundError(f"The file '{obj_file}' does not exist.")

        self.file_name = os.path.splitext(os.path.basename(obj_file))[0]
        self.working_dir = os.path.join(WORK_DIR, self.file_name)
        os.makedirs(self.working_dir, exist_ok=True)
        self.obj_file = obj_file

    def estimate_best_alpha(self, pcd, alpha_values=None, visualize=False):
        """
        Estimates the best alpha value by testing multiple candidates.
        Chooses the one that gives a smooth but dense mesh.
        """
        if alpha_values is None:
            alpha_values = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]

        best_alpha = None
        best_density = 0

        print("\n[INFO] Estimating best alpha value...")
        for alpha in alpha_values:
            try:
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
                num_vertices = np.asarray(mesh.vertices).shape[0]
                num_faces = np.asarray(mesh.triangles).shape[0]
                density = num_faces / (num_vertices + 1e-6)
                print(f"  α={alpha:.3f} → vertices={num_vertices}, faces={num_faces}, density={density:.2f}")

                if density > best_density:
                    best_density = density
                    best_alpha = alpha

            except Exception as e:
                print(f"  α={alpha:.3f} failed: {e}")

        if best_alpha is None:
            raise ValueError("Could not find a valid alpha value.")

        print(f"[INFO] ✅ Best alpha automatically chosen: {best_alpha:.3f}")
        return best_alpha

    def alpha_shape_filter_auto(self, visualize=False):
        """
        Automatically removes surrounding noise using Alpha Shape boundary filtering.
        Automatically finds the best alpha value.
        """
        input_path = self.obj_file
        print("\n[INFO] Loading point cloud...")

        # Properly load .obj mesh as a point cloud
        if input_path.lower().endswith(".obj"):
            mesh = o3d.io.read_triangle_mesh(input_path)
            if mesh.is_empty():
                print("Error: mesh is empty.")
                return None
            pcd = mesh.sample_points_uniformly(number_of_points=200000)
        else:
            pcd = o3d.io.read_point_cloud(input_path)

        if len(pcd.points) == 0:
            print("Error: point cloud is empty.")
            return None

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

        # Step 1: Estimate best alpha
        best_alpha = self.estimate_best_alpha(pcd)

        # Step 2: Generate alpha shape mesh
        print("[INFO] Generating alpha shape mesh with best alpha...")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, best_alpha)

        # Step 3: Filter points inside the alpha shape
        sampled = mesh.sample_points_uniformly(number_of_points=len(pcd.points))
        sampled_tree = o3d.geometry.KDTreeFlann(sampled)

        inside_points = []
        for i, point in enumerate(pcd.points):
            [k, idx, _] = sampled_tree.search_knn_vector_3d(point, 1)
            if k > 0:
                inside_points.append(i)

        filtered = pcd.select_by_index(inside_points)

        if visualize:
            print("[INFO] Visualizing result...")
            o3d.visualization.draw_geometries([filtered])

        # ✅ Save in the working directory instead of input path

        output_path = os.path.join(self.working_dir, "alpha_filtered_auto.ply")
        o3d.io.write_point_cloud(output_path, filtered)
        print(f"[DONE] Auto Alpha shape filtered file saved at: {output_path}")
        return output_path
