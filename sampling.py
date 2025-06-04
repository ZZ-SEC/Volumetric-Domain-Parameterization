import numba
import numpy as np
import trimesh
import trimesh.proximity
from utils import get_uvw_bound, get_uvw


@numba.njit()
def fill_search_3d(mask, start_pos, value=-1):
    shape = np.array(mask.shape, dtype=np.int32)
    A, B, C = shape
    around = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    Q = np.empty((A * B * C, 3), dtype=np.int32)
    Q[0, :] = start_pos
    Q_s = 0
    Q_len = 1
    mask[start_pos[0], start_pos[1], start_pos[2]] = value
    while Q_len > 0:
        pos = Q[Q_s, :]
        Q_s += 1
        Q_len -= 1
        for rel in around:
            pos2 = pos + rel
            if np.all(pos2 >= 0) and np.all(pos2 < shape):
                if mask[pos2[0], pos2[1], pos2[2]] == 1:
                    Q[Q_s + Q_len] = pos2
                    Q_len += 1
                    mask[pos2[0], pos2[1], pos2[2]] = value
    return mask


def InteriorSample(V, F, N, expand=0):
    mesh = trimesh.Trimesh(V, F)
    volumn = mesh.volume
    h = (volumn / N) ** (1 / 3)
    min = np.min(V, axis=0)
    max = np.max(V, axis=0)
    cen = (min + max) / 2
    dis = max - min
    min, max = cen - dis / 2 * 1.5, cen + dis / 2 * 1.5
    num = np.ceil((max - min) / h).astype(np.int32)
    grid = get_uvw(num[0], num[1], num[2]) * (num * h).reshape([1, 1, 3]) + min.reshape([1, 1, 3])
    # mark bound as 0
    mask = np.ones([num[0], num[1], num[2]], dtype=np.int32)
    max_len = np.max(
        np.linalg.norm(np.concatenate([V[F[:, 0], :] - V[F[:, 1], :], V[F[:, 1], :] - V[F[:, 2], :], V[F[:, 0], :] - V[F[:, 2], :]]), axis=1))
    N_subdivide = int(np.max([np.ceil(np.log2(max_len / h)), 0]))
    for i in range(N_subdivide):
        mesh = mesh.subdivide()
    V_sub = mesh.vertices
    idx_bound = np.floor((V_sub - min) / h).astype(np.int32)
    idx_valid = np.argwhere(np.all((idx_bound < num.reshape([1, 3]) - 1) & (idx_bound >= 1), axis=1))[:, 0]
    idx_bound = idx_bound[idx_valid, :]
    around = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    for i in range(expand):
        idx_bound_expand = (idx_bound.reshape([-1, 1, 3]) + around.reshape([1, 6, 3])).reshape([-1, 3])
        idx_bound = np.unique(np.concatenate([idx_bound_expand, idx_bound]), axis=0)
        idx_valid = np.argwhere(np.all((idx_bound < num.reshape([1, 3]) - 1) & (idx_bound >= 1), axis=1))[:, 0]
        idx_bound = idx_bound[idx_valid, :]
    mask[idx_bound[:, 0], idx_bound[:, 1], idx_bound[:, 2]] = 0
    # fill the outside as -1
    fill_search_3d(mask, np.array([0, 0, 0], dtype=np.int32), -1)
    vertices = get_uvw_bound(num[0] + 1, num[1] + 1, num[2] + 1) * (num * h).reshape([1, 1, 3]) + min.reshape([1, 1, 3])
    is_inner_cube = (mask >= 0)
    cube_inner_idx = np.argwhere(is_inner_cube)
    shift = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.int32).reshape([1, 8, 3])
    vertex_inner_idx = cube_inner_idx.reshape([-1, 1, 3]) + shift
    vertex_inner_idx_flatten = vertex_inner_idx.reshape([-1, 3])
    vertex_inner_idx_uniq = np.unique(vertex_inner_idx_flatten, axis=0)
    vertices_inner = vertices[vertex_inner_idx_uniq[:, 0], vertex_inner_idx_uniq[:, 1], vertex_inner_idx_uniq[:, 2]]
    vertices_idx_final = -np.ones([num[0] + 1, num[1] + 1, num[2] + 1], dtype=np.int32)
    vertices_idx_final[vertex_inner_idx_uniq[:, 0], vertex_inner_idx_uniq[:, 1], vertex_inner_idx_uniq[:, 2]] = \
        np.arange(0, vertex_inner_idx_uniq.shape[0])
    cube_vertices_idx = vertices_idx_final[vertex_inner_idx[:, :, 0], vertex_inner_idx[:, :, 1], vertex_inner_idx[:, :, 2]]
    shift_hes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1],
                          [-1, -1, 0], [-1, 0, -1], [0, -1, -1],
                          [-1, 1, 0], [-1, 0, 1], [0, -1, 1],
                          [1, -1, 0], [1, 0, -1], [0, 1, -1],
                          [1, 1, 0], [1, 0, 1], [0, 1, 1]
                          ], dtype=np.int32).reshape([1, 19, 3])
    vertex_hes = cube_inner_idx.reshape([-1, 1, 3]) + shift_hes
    vertex_hes = vertices_idx_final[vertex_hes[:, :, 0], vertex_hes[:, :, 1], vertex_hes[:, :, 2]]
    valid = np.all(vertex_hes >= 0, axis=1)
    vertex_hes = vertex_hes[valid, :]
    return vertices_inner, cube_vertices_idx, vertex_hes, h
