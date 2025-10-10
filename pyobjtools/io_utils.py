def read_obj(filename):
    obj_meta_data = ''
    obj_data = []
    v_id = 0
    usemtl = ''
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('usemtl'):
                usemtl = line

            if line.startswith('v') or line.startswith('vt'):
                obj_data.append(tuple([v_id := v_id + 1] + line.split()))
            elif line.startswith('f'):
                data_tuple = [v_id := v_id + 1] + line.split()
                if usemtl != '':
                    data_tuple.append(usemtl)
                obj_data.append(tuple(data_tuple))
            elif not line.startswith('usemtl'):
                obj_meta_data += line

    return [obj_meta_data, obj_data]


def write_obj(obj_meta_data, obj_data, filename):
    with open(filename, 'w') as f:
        f.write(obj_meta_data)
        last_usemtl = ''
        for obj in obj_data:
            if obj[1] == 'f' and obj[-1].startswith('usemtl'):
                mtl = obj[-1]
                if last_usemtl != mtl:
                    last_usemtl = mtl
                    f.write(last_usemtl)

                f.write(' '.join(map(str, obj[1:-1])) + '\n')
            else:
                f.write(' '.join(map(str, obj[1:])) + '\n')


if __name__ == '__main__':
    md, od = read_obj('samples/cube.obj')
    print(md, '\n', od)
    write_obj(md, od, 'writen_cube.obj')
