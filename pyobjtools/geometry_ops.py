def remove_vertices_by_vertexId(obj_data: list, removed_vertice_ids: list | set):
    removed_vertice_ids = set(removed_vertice_ids)
    removed_idx = set()
    mx_vertices_num = 0
    map_numbers = {}

    for idx, obj in enumerate(obj_data):
        if obj[1] == 'v':
            if obj[0] in removed_vertice_ids:
                removed_idx.add(idx)
            else:
                mx_vertices_num += 1
                map_numbers[idx + 1] = mx_vertices_num
        elif obj[1] == 'f':
            for part in obj[2:]:
                v_id = part.split('/')[0]
                if int(v_id) in removed_vertice_ids:
                    removed_idx.add(idx)
                    break

    print(map_numbers)
    return apply_remove(obj_data, removed_idx, map_numbers)


def apply_swap_dict(vertex: tuple | list, swap_dict: dict):
    tmp = list(vertex)
    for i in range(2, len(tmp)):
        x = int(tmp[i])
        tmp[i] = swap_dict[x]
    return tmp


def apply_remove(obj_data: list, remove_indices: set, swap_dict: dict):
    new_obj_data = []
    cnt = 1
    for idx, obj in enumerate(obj_data):
        if idx not in remove_indices:
            tmp = list(obj)
            tmp[0] = cnt

            if tmp[1] == 'f':
                tmp = apply_swap_dict(tmp, swap_dict)

            new_obj_data.append(tmp)
            cnt += 1

    return new_obj_data
