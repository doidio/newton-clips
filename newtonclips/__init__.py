import numpy as np

import renderer


def tri_surface_from_tet(tetrahedrons: np.ndarray) -> np.ndarray:
    # 生成所有四面体的四个面，每个面顶点排序
    faces_indices = np.array([[0, 2, 1], [1, 2, 3], [0, 1, 3], [0, 3, 2]])
    faces = tetrahedrons[:, faces_indices].reshape(-1, 3)
    sorted_faces = np.sort(faces, axis=1)

    # 按字典序排序所有面
    sort_order = np.lexsort((sorted_faces[:, 2], sorted_faces[:, 1], sorted_faces[:, 0]))
    sorted_faces = sorted_faces[sort_order]

    # 确定各面组及其出现次数
    diff = np.ones(len(sorted_faces), dtype=bool)
    diff[1:] = np.any(sorted_faces[1:] != sorted_faces[:-1], axis=1)
    group_starts = np.where(diff)[0]
    group_ends = np.append(group_starts[1:], len(sorted_faces))
    group_lengths = group_ends - group_starts

    # 提取仅出现一次的面
    surface_mask = (group_lengths == 1)
    surface_faces = sorted_faces[group_starts[surface_mask]]

    return surface_faces
