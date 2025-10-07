import os

from . import csv_utils
from . import geometry_ops
from . import io_utils


class ObjFile:
    def __init__(self, obj_file_path):
        if obj_file_path.endswith('.obj'):
            self.obj_file_path = obj_file_path
            self.meta_data, self.obj_data = io_utils.read_obj(self.obj_file_path)
        else:
            self.obj_file_path = obj_file_path
            self.meta_data, self.obj_data = csv_utils.read_cv2_to_obj(self.obj_file_path)

    def convert_to_csv(self, output_path='../outputs'):
        file_name = os.path.splitext(os.path.basename(self.obj_file_path))[0]
        csv_utils.write_object_to_csv(self.obj_data, self.obj_file_path, file_name, output_path)

    def write_obj_file(self, write_path):
        io_utils.write_obj(self.meta_data, self.obj_data, write_path)

    def remove_vertices_list_by_ids(self, removed_vertice_ids: list | set):
        self.obj_data = geometry_ops.remove_vertices_by_vertexId(self.obj_data, removed_vertice_ids)