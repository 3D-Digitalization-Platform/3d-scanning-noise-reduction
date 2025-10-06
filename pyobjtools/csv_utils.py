import csv
import os

import io_utils


def create_object_folder(base_path, object_name):
    folder_path = os.path.join(base_path, object_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def write_object_to_csv(obj_meta_data, obj_data, filename):
    path = create_object_folder('../outputs', filename)

    with open(os.path.join(path, 'meta.txt'), 'w') as f:
        f.write(obj_meta_data)

    vertices = []
    textures = []
    faces = []

    for data in obj_data:
        if data[1] == 'v':
                vertices.append(data)
        elif data[1] == 'vt':
                textures.append(data)
        elif data[1] == 'f':
            faces.append(data)

    with open(os.path.join(path, 'vertices.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vertices)

    with open(os.path.join(path, 'textures.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(textures)

    with open(os.path.join(path, 'faces.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(faces)


if __name__ == '__main__':
    md, od = io_utils.read_obj('samples/cube.obj')
    write_object_to_csv(md, od, 'cube')
