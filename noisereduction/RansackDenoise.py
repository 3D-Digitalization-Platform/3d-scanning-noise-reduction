import csv
import os

import open3d as o3d
import pandas as pd

from pyobjtools import ObjFile

WORK_DIR = './temp/RANSACK'


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


if __name__ == '__main__':
    noiseModel = RansackDenoise('./temp/HDBSCAN/texturedMesh/texturedMesh_noise_reduced.obj')
    print(noiseModel.run_ransack())
