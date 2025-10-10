import csv
import os

from pyobjtools import ObjFile

def main():
    obj = ObjFile('samples/texturedMesh.obj')  # load .obj file

    removeList = set()
    for file in os.listdir('samples/texturedMeshNoise/clusters'):
        filePath = os.path.join('samples/texturedMeshNoise/clusters', file)
        for line in csv.reader(open(filePath)):
            removeList.add(int(line[0]))
    print(len(removeList))

    obj.remove_vertices_list_by_ids(removeList)
    obj.write_obj_file('removed_texturedMesh.obj')


def mainCube():
    obj = ObjFile('samples/cube.obj')
    removeList = [2,3]
    obj.remove_vertices_list_by_ids(removeList)
    obj.write_obj_file('removed_cube.obj')


if __name__ == '__main__':
    main()
