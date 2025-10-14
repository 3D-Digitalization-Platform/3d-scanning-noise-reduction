import os
import hdbscan
import pandas as pd
import csv
from pyobjtools import ObjFile

WORK_DIR = './temp/HDBSCAN'


class HdbscanDenoise:
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

    def run_hdbscan(self):
        # Load vertex data
        vertices_filename = self.vertices_path
        df = pd.read_csv(vertices_filename, header=None, names=["VertexID", "VertexType", "X", "Y", "Z"])

        # Prepare points for clustering
        points = df[['X', 'Y', 'Z']].values

        # Run HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
        labels = clusterer.fit_predict(points)
        df['Cluster'] = labels

        # Create output directory
        output_dir = f"{self.working_dir}/clusters/{self.file_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Identify largest cluster
        cluster_sizes = df['Cluster'].value_counts()
        largest_cluster_id = cluster_sizes.idxmax()

        # Separate largest cluster (data) and others (noise)
        data_df = df[df['Cluster'] == largest_cluster_id].drop(columns=['Cluster'])
        noise_df = df[df['Cluster'] != largest_cluster_id].drop(columns=['Cluster'])

        # Save as CSV without headers
        data_path = f"{output_dir}/data.csv"
        noise_path = f"{output_dir}/noise.csv"
        data_df.to_csv(data_path, index=False, header=False)
        noise_df.to_csv(noise_path, index=False, header=False)

        print("Clustering completed.")
        print(f"Data saved to:\n- {data_path}\n- {noise_path}")

        remove_set = {int(line[0]) for line in csv.reader(open(noise_path))}

        self.obj.remove_vertices_list_by_ids(remove_set)

        output_obj = f'{self.working_dir}/{self.file_name}_noise_reduced.obj'
        self.obj.write_obj_file(output_obj)
        print(f"Noise reduced OBJ saved as: {output_obj}")

        return noise_path


if __name__ == '__main__':
    tmp = HdbscanDenoise('./texturedMesh.obj')
    tmp.run_hdbscan()
