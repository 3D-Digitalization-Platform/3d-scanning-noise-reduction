def read_obj(filename):
    obj_meta_data = ''
    obj_data = []
    v_id = 0
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v') or line.startswith('vt') or line.startswith('f'):
                obj_data.append(tuple([v_id := v_id + 1] + line.split()))
            else:
                obj_meta_data += line

    return [obj_meta_data, obj_data]


def write_obj(obj_meta_data, obj_data, filename):
    with open(filename, 'w') as f:
        f.write(obj_meta_data)
        f.write('\n')
        for obj in obj_data:
            f.write(' '.join(map(str, obj[1:])) + '\n')


if __name__ == '__main__':
    md, od = read_obj('mm.obj')
    print(md, '\n', od)
    write_obj(md, od, 'mmw.obj')
