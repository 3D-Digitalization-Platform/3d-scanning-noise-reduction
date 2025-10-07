import csv
import os

import io_utils


def create_object_folder(base_path, object_name):
    folder_path = os.path.join(base_path, object_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def write_object_to_csv(obj_meta_data, obj_data, filename,base_path='../outputs'):
    path = create_object_folder(base_path, filename)

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


def read_cv2_to_obj(folder_path):
    obj_meta_data = ''
    obj_data = []

    required_files = [
        'meta.txt',
        'vertices.csv',
        'textures.csv',
        'faces.csv'
    ]

    # check all files exist
    missing = [f for f in required_files if not os.path.exists(os.path.join(folder_path, f))]
    if missing:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing)} in {folder_path}")

    meta_path = os.path.join(folder_path, 'meta.txt')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            obj_meta_data = f.read()

    vertices_path = os.path.join(folder_path, 'vertices.csv')
    if os.path.exists(vertices_path):
        with open(vertices_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                row[0] = int(row[0])
                obj_data.append(tuple(row))

    textures_path = os.path.join(folder_path, 'textures.csv')
    if os.path.exists(textures_path):
        with open(textures_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                row[0] = int(row[0])
                obj_data.append(tuple(row))

    faces_path = os.path.join(folder_path, 'faces.csv')
    if os.path.exists(faces_path):
        with open(faces_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                row[0] = int(row[0])
                obj_data.append(tuple(row))

    return obj_meta_data, obj_data


if __name__ == '__main__':
    md, od = io_utils.read_obj('samples/cube.obj')
    write_object_to_csv(md, od, 'cube')

    md_read, od_read = read_cv2_to_obj('../outputs/cube')
    print(md_read, od_read)