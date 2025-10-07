from pyobjtools import ObjFile

if __name__ == '__main__':
    obj = ObjFile('samples/cube.obj')  # load .obj file
    # obj = ObjFile('outputs/cube')  # load from csv folder format

    # list of vertices to remove
    obj.remove_vertices_list_by_ids([2])

    obj.write_obj_file('removed_cube.obj')  # save as .obj file

    # obj.convert_to_csv('removed_cube')  # convert to csv folder format
