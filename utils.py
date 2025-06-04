import queue
import pymeshlab as ml
import numba
import numpy as np
import torch
import random
import pyvista as pv
import openmesh as OM
from scipy.interpolate import RBFInterpolator
from training_manager import TrainingManager


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def penalty_pos(x, k=1):
    return torch.clip(-x, min=0) * k


def get_uvw_bound(A, B=None, C=None):
    # M[i,j,k]=(ih,jh,kh)
    if B is None:
        B, C = A, A
    hx, hy, hz = 1.0 / (A - 1), 1.0 / (B - 1), 1.0 / (C - 1)
    Mat = np.zeros((A, B, C, 3), dtype=np.float64)
    for i in range(A):
        Mat[i, :, :, 0] = i * hx
    for i in range(B):
        Mat[:, i, :, 1] = i * hy
    for i in range(C):
        Mat[:, :, i, 2] = i * hz
    return Mat


def get_uvw(A, B=None, C=None):
    # M[i,j,k,:]=((i+0.5)h,(j+0.5)h,(k+0.5)h)
    if B is None:
        B, C = A, A
    hx, hy, hz = 1.0 / A, 1.0 / B, 1.0 / C
    Mat = np.zeros([A, B, C, 3])
    for i in range(A):
        Mat[i, :, :, 0] = (i + 0.5) * hx
    for i in range(B):
        Mat[:, i, :, 1] = (i + 0.5) * hy
    for i in range(C):
        Mat[:, :, i, 2] = (i + 0.5) * hz
    return Mat


def draw_tri_mesh(pl, bound, cube=False, update=None):
    faces = bound.faces
    n_faces = faces.shape[0]
    faces = np.concatenate([np.ones([n_faces, 1], dtype=np.int32) * 3, faces], axis=1)
    if update is not None:
        mesh = update
        mesh.points = bound.vertices
    else:
        if cube:
            vertices_c = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
            faces_c = np.array([[4, 0, 3, 2, 1], [4, 4, 5, 6, 7], [4, 0, 1, 5, 4], [4, 3, 7, 6, 2], [4, 0, 4, 7, 3], [4, 1, 2, 6, 5]], dtype=np.int32)
            mesh_c = pv.PolyData(vertices_c, faces_c)
            pl.add_mesh(mesh_c, color="r", line_width=2, style="wireframe", name="cube")
        mesh = pv.PolyData(bound.vertices, faces)
        pl.add_mesh(mesh, color="gray", edge_color="k", line_width=2, opacity=1, show_edges=True)
        pl.reset_camera()
    return mesh


@numba.njit()
def get_surface_triangles(A, B, C):
    N_points = 2 * (A * B + B * C + A * C) - 4 * (A + B + C) + 8
    V_idx = np.zeros((N_points, 3), dtype=np.int32)
    V_mapper = np.zeros((A, B, C), dtype=np.int32)
    count = 0
    for i in range(A):
        for j in range(B):
            # i,j,C
            V_idx[count:count + 2, :] = np.array(((i, j, 0), (i, j, C - 1)), dtype=np.int32)
            V_mapper[i, j, 0], V_mapper[i, j, -1] = count, count + 1
            count += 2
    for i in range(B):
        for j in range(1, C - 1):
            # A,i,j
            V_idx[count:count + 2, :] = np.array(((0, i, j), (A - 1, i, j)), dtype=np.int32)
            V_mapper[0, i, j], V_mapper[-1, i, j] = count, count + 1
            count += 2
    for i in range(1, A - 1):
        for j in range(1, C - 1):
            # i,0/B,j
            V_idx[count:count + 2, :] = np.array(((i, 0, j), (i, B - 1, j)), dtype=np.int32)
            V_mapper[i, 0, j], V_mapper[i, -1, j] = count, count + 1
            count += 2
    tri0 = np.zeros((B - 1, C - 1, 2, 3), dtype=np.int32)
    tri3 = np.zeros((B - 1, C - 1, 2, 3), dtype=np.int32)
    for i in range(B - 1):
        for j in range(C - 1):
            tri0[i, j, 0, 0] = V_mapper[0, i, j]
            tri0[i, j, 0, 1] = V_mapper[0, i, j + 1]
            tri0[i, j, 0, 2] = V_mapper[0, i + 1, j]
            tri0[i, j, 1, 0] = V_mapper[0, i + 1, j]
            tri0[i, j, 1, 1] = V_mapper[0, i, j + 1]
            tri0[i, j, 1, 2] = V_mapper[0, i + 1, j + 1]
            tri3[i, j, 0, 0] = V_mapper[-1, i, j]
            tri3[i, j, 0, 1] = V_mapper[-1, i + 1, j]
            tri3[i, j, 0, 2] = V_mapper[-1, i, j + 1]
            tri3[i, j, 1, 0] = V_mapper[-1, i + 1, j]
            tri3[i, j, 1, 1] = V_mapper[-1, i + 1, j + 1]
            tri3[i, j, 1, 2] = V_mapper[-1, i, j + 1]
    tri1 = np.zeros((A - 1, C - 1, 2, 3), dtype=np.int32)
    tri4 = np.zeros((A - 1, C - 1, 2, 3), dtype=np.int32)
    for i in range(A - 1):
        for j in range(C - 1):
            tri1[i, j, 0, 0] = V_mapper[i, 0, j]
            tri1[i, j, 0, 1] = V_mapper[i + 1, 0, j]
            tri1[i, j, 0, 2] = V_mapper[i, 0, j + 1]
            tri1[i, j, 1, 0] = V_mapper[i + 1, 0, j]
            tri1[i, j, 1, 1] = V_mapper[i + 1, 0, j + 1]
            tri1[i, j, 1, 2] = V_mapper[i, 0, j + 1]
            tri4[i, j, 0, 0] = V_mapper[i, -1, j]
            tri4[i, j, 0, 1] = V_mapper[i, -1, j + 1]
            tri4[i, j, 0, 2] = V_mapper[i + 1, -1, j]
            tri4[i, j, 1, 0] = V_mapper[i + 1, -1, j]
            tri4[i, j, 1, 1] = V_mapper[i, -1, j + 1]
            tri4[i, j, 1, 2] = V_mapper[i + 1, -1, j + 1]
    # face z=0,z=1
    tri2 = np.zeros((A - 1, B - 1, 2, 3), dtype=np.int32)
    tri5 = np.zeros((A - 1, B - 1, 2, 3), dtype=np.int32)
    for i in range(A - 1):
        for j in range(B - 1):
            tri2[i, j, 0, 0] = V_mapper[i, j, 0]
            tri2[i, j, 0, 1] = V_mapper[i, j + 1, 0]
            tri2[i, j, 0, 2] = V_mapper[i + 1, j, 0]
            tri2[i, j, 1, 0] = V_mapper[i + 1, j, 0]
            tri2[i, j, 1, 1] = V_mapper[i, j + 1, 0]
            tri2[i, j, 1, 2] = V_mapper[i + 1, j + 1, 0]
            tri5[i, j, 0, 0] = V_mapper[i, j, -1]
            tri5[i, j, 0, 1] = V_mapper[i + 1, j, -1]
            tri5[i, j, 0, 2] = V_mapper[i, j + 1, -1]
            tri5[i, j, 1, 0] = V_mapper[i + 1, j, -1]
            tri5[i, j, 1, 1] = V_mapper[i + 1, j + 1, -1]
            tri5[i, j, 1, 2] = V_mapper[i, j + 1, -1]
    tri = [tri0, tri1, tri2, tri3, tri4, tri5]
    tri = [trii.reshape((-1, 3)) for trii in tri]
    return V_idx, tri


def pv_reset_plotter(pl):
    actors = []
    for act in pl.actors:
        actors.append(act)
    for act in actors:
        pl.remove_actor(act)


def get_tar_cube_torch(bound_):
    with torch.no_grad():
        N = bound_.shape[0]
        bound_clamp = torch.clamp(bound_.detach(), 0, 1)
        temp = torch.cat([bound_clamp, 1 - bound_clamp], dim=1)
        idx_min = torch.argmin(temp, dim=1)
        bound_clamp[torch.arange(N), idx_min % 3] = 1.0 * (idx_min >= 3)
        return bound_clamp


def get_tar_cube_faceid_torch(bound_):
    with torch.no_grad():
        bound_clamp = torch.clamp(bound_.detach(), 0, 1)
        temp = torch.cat([bound_clamp, 1 - bound_clamp], dim=1)
        idx_min = torch.argmin(temp, dim=1)
        return idx_min.detach()


def get_tar_cube_norm_torch(bound_):
    with torch.no_grad():
        N = bound_.shape[0]
        idx_min = get_tar_cube_faceid_torch(bound_)
        ret = torch.zeros_like(bound_)
        ret[torch.arange(N), idx_min % 3] = 2.0 * (idx_min >= 3) - 1
        return ret


def get_tar_cube_star_torch(bound_):
    with torch.no_grad():
        bound_c = bound_ - 0.5
        scale = 0.5 / torch.max(torch.abs(bound_c), dim=-1)[0]
        bound_tar = bound_c * scale.unsqueeze(-1) + 0.5
        return bound_tar.detach()


def get_tar_cube_norm_star_torch(bound_):
    return get_tar_cube_norm_torch(get_tar_cube_star_torch(bound_))


def get_tar_ball_torch(bound_):
    bound = bound_.detach().clone()
    len = torch.norm(bound, dim=1)
    bound = bound / len.reshape([-1, 1])
    return bound


def get_tar_semi_cube(bound_, d=1):
    V = 1 + 6 * d + 3 * np.pi * d * d + 4 * np.pi / 3 * d ** 3
    scale = V ** (1 / 3)
    bound_c = (bound_.detach() - 0.5) * scale
    bound = bound_c + 0.5
    bound_tar = bound.clone()
    bound_tar_norm = bound.clone()

    out_cube = torch.abs(bound_c) > 0.5
    any_out = torch.any(out_cube, dim=1)
    inner = torch.argwhere(torch.logical_not(any_out))[:, 0]
    any_out = torch.argwhere(any_out)[:, 0]
    bound_out = bound[any_out, :]
    ball_center = torch.clip(bound_out, 0, 1)
    bound_out_center = bound_out - ball_center
    temp = get_tar_ball_torch(bound_out_center)
    bound_tar[any_out, :] = temp * d + ball_center
    bound_tar_norm[any_out, :] = temp
    bound_tar[inner, :] = (get_tar_cube_torch((bound_c[inner, :]) / (1 + 2 * d) + 0.5) - 0.5) * (1 + 2 * d) + 0.5
    bound_tar_norm[inner, :] = get_tar_cube_norm_torch(bound[inner, :])
    bound_tar = (bound_tar - 0.5) / scale + 0.5
    return bound_tar, bound_tar_norm


def sel_corners_dis(out_bound):
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
    vec_corners = corners - 0.5
    idx_corners = np.argmax(np.sum(out_bound[:, None, :] * vec_corners[None, :, :], axis=-1), axis=0)
    return idx_corners


def dist_plane(points, plane_n, plane_v):
    shape = points.shape
    points = points.reshape([-1, 3])
    ret = np.sum(points * plane_n[None, :], axis=1) - plane_v
    if len(shape) == 1:
        return ret[0]
    return ret.reshape(shape[:-1])


def sel_edges(V, bound, idx_corners):
    V_mesh, F = bound.vertices, bound.faces
    coord_corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
    rbf = RBFInterpolator(V[idx_corners, :], coord_corners - V[idx_corners, :], kernel='linear')
    V += rbf(V)
    edges_local = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [5, 6], [6, 7], [7, 4]], dtype=np.int32)
    edges_vertex_mesh_idx = []
    edges_vertex_tar = []
    V_prop = -np.ones([V.shape[0]], dtype=np.int32)
    F_prop = -np.ones([F.shape[0]], dtype=np.int32)
    V_prop[idx_corners] = np.arange(8) + 18
    for edge_idx in range(12):
        edge_local = edges_local[edge_idx, :]
        edge_global = idx_corners[edge_local]
        coord_edge = coord_corners[edge_local, :]
        edge_face_axis = np.argwhere(np.abs(coord_edge[0, :] - coord_edge[1, :]) < 1e-6)[:, 0]
        edge_axis_free = 3 - edge_face_axis[0] - edge_face_axis[1]
        increase = (coord_edge[0, edge_axis_free] < coord_edge[1, edge_axis_free])
        edge_face_value = coord_edge[0, edge_face_axis]
        edge_faceid = np.round(edge_face_axis + edge_face_value * 3).astype(np.int32)
        edge_face_norm = np.zeros([2, 3], dtype=np.float32)
        edge_face_norm[[0, 1], edge_face_axis] = 1 - 2 * edge_face_value
        if np.linalg.norm(np.cross(edge_face_norm[0, :], edge_face_norm[1, :]) - (coord_edge[1, :] - coord_edge[0, :])) > 0.1:
            faceid_l, faceid_r = edge_faceid[1], edge_faceid[0]
        else:
            faceid_l, faceid_r = edge_faceid[0], edge_faceid[1]

        cutplane_n = np.cross(coord_edge[1, :] - coord_edge[0, :], 0.5 - coord_edge[0, :])
        cutplane_n /= np.linalg.norm(cutplane_n)
        cutplane_v = np.sum(0.5 * cutplane_n)

        f_list = []
        e_list = []
        head_idx = edge_global[0]
        # near_fid=VF[head_idx,:VF_N[head_idx]]
        near_fid = np.argwhere(F == head_idx)[:, 0]
        near_f = F[near_fid, :]
        N_near = len(near_fid)
        eid_local_round = np.argwhere(near_f == head_idx)[:, 1]
        e_round = near_f[
            np.column_stack([np.arange(N_near), np.arange(N_near)]), np.column_stack([(eid_local_round + 1) % 3, (eid_local_round + 2) % 3])]
        e_round_side = dist_plane(V[e_round, :], cutplane_n, cutplane_v) > 0
        idx_cross = np.argwhere(np.sum(e_round_side, axis=1) == 1)[:, 0]
        idx_idx = np.argmin(np.min(np.linalg.norm(V[e_round[idx_cross, :], :] - coord_edge[None, None, 1, :], axis=-1), axis=-1))
        idx_first = idx_cross[idx_idx]
        e_cross = eid_local_round[idx_first]
        e_cross_fid = near_fid[idx_first]
        e_list.append(e_cross)
        f_list.append(e_cross_fid)

        while True:
            current_fid = f_list[-1]
            current_eid_local = e_list[-1]
            current_f = F[current_fid, :]
            current_e_local = [(current_eid_local + 1) % 3, (current_eid_local + 2) % 3]
            current_e = current_f[current_e_local]
            fid_contains_e0 = np.argwhere(F == current_e[0])[:, 0]
            idx_fid_contains_e = np.argwhere(F[fid_contains_e0, :] == current_e[1])[:, 0]
            fid_contains_e = fid_contains_e0[idx_fid_contains_e]
            next_fid = np.sum(fid_contains_e) - current_fid
            f_list.append(next_fid)
            next_f = F[next_fid, :]
            next_vid = np.sum(next_f) - np.sum(current_e)
            if next_vid == edge_global[1]:
                break
            current_e_side = dist_plane(V[current_e, :], cutplane_n, cutplane_v) > 0
            next_v_side = dist_plane(V[next_vid, :], cutplane_n, cutplane_v) > 0
            current_e0_next_f_local = np.argwhere(next_f == current_e[0])[0, 0]
            current_e1_next_f_local = np.argwhere(next_f == current_e[1])[0, 0]
            if next_v_side == current_e_side[0]:
                e_list.append(current_e0_next_f_local)
            else:
                e_list.append(current_e1_next_f_local)

        Ne = len(e_list)
        edge_vertex_mesh = np.zeros([Ne + 2, 3], dtype=np.float32)
        edge_vertex_map = np.zeros([Ne + 2, 3], dtype=np.float32)
        edge_vertex_mesh[[0, -1], :] = V_mesh[edge_global, :]
        edge_vertex_map[[0, -1], :] = V[edge_global, :]
        for i in range(Ne):
            f = F[f_list[i], :]
            f_eid_local = e_list[i]
            f_e_local = [(f_eid_local + 1) % 3, (f_eid_local + 2) % 3]
            e = f[f_e_local]
            coord_e_cube = V[e, :]
            coord_e_mesh = V_mesh[e, :]
            e_dist_plane = np.abs(dist_plane(coord_e_cube, cutplane_n, cutplane_v))
            weight = e_dist_plane[[1, 0]] / max(e_dist_plane[0] + e_dist_plane[1], 1e-7)
            weight[1] = 1 - weight[0]
            edge_vertex_mesh[i + 1, :] = np.sum(coord_e_mesh * weight[:, None], axis=0)
            edge_vertex_map[i + 1, :] = np.sum(coord_e_cube * weight[:, None], axis=0)
        len_tar = np.linalg.norm(edge_vertex_map[1:, :] - edge_vertex_map[:-1, :], axis=1)
        len_tar = len_tar / np.sum(len_tar)
        param = np.concatenate([np.array([0], dtype=len_tar.dtype), np.cumsum(len_tar)])
        if not increase:
            param = 1 - param
        edge_vertex_tar = edge_vertex_map.copy()
        edge_vertex_tar[:, edge_face_axis] = edge_face_value[None, :]
        edge_vertex_tar[:, edge_axis_free] = param
        Nv = V_mesh.shape[0]
        edges_vertex_mesh_idx.append(np.concatenate([edge_global[0:1], np.arange(Ne) + Nv, edge_global[1:2]]))
        edges_vertex_tar.append(edge_vertex_tar)
        V_mesh = np.concatenate([V_mesh, edge_vertex_mesh[1:-1, :]])
        V = np.concatenate([V, edge_vertex_map[1:-1, :]])
        V_prop_append = np.zeros([Ne], dtype=np.int32)
        V_prop_append[:] = 6 + edge_idx
        V_prop = np.concatenate([V_prop, V_prop_append])
        F_append = []
        F_prop_append = []
        v1 = edge_global[0]
        v2 = F[f_list[0], (e_list[0] + 1) % 3]
        v3 = F[f_list[0], (e_list[0] + 2) % 3]
        v4 = Nv
        F_append.append([v1, v2, v4])
        F_append.append([v1, v4, v3])
        F_prop_append.append(faceid_r)
        F_prop_append.append(faceid_l)
        v1 = edge_global[1]
        v1_idx_local = np.argwhere(F[f_list[-1], :] == v1)[0, 0]
        v2 = F[f_list[-1], (v1_idx_local + 1) % 3]
        v3 = F[f_list[-1], (v1_idx_local + 2) % 3]
        v4 = Nv + Ne - 1
        F_append.append([v1, v2, v4])
        F_append.append([v1, v4, v3])
        F_prop_append.append(faceid_l)
        F_prop_append.append(faceid_r)
        for i in range(Ne - 1):
            f = F[f_list[i + 1], :]
            e_last = F[f_list[i], [(e_list[i] + 1) % 3, (e_list[i] + 2) % 3]]
            v1 = e_last[1]
            v2 = e_last[0]
            v3 = np.sum(f) - v1 - v2
            v4 = Nv + i
            v5 = Nv + i + 1
            if f[e_list[i + 1]] == v2:
                F_append.append([v1, v4, v5])
                F_append.append([v2, v5, v4])
                F_append.append([v2, v3, v5])
                F_prop_append.append(faceid_l)
                F_prop_append.append(faceid_r)
                F_prop_append.append(faceid_r)
            else:
                F_append.append([v1, v4, v3])
                F_append.append([v2, v5, v4])
                F_append.append([v3, v4, v5])
                F_prop_append.append(faceid_l)
                F_prop_append.append(faceid_r)
                F_prop_append.append(faceid_l)
        F_reamin = np.ones([F.shape[0]], dtype=bool)
        F_reamin[f_list] = False
        idx_reamin = np.argwhere(F_reamin)[:, 0]
        F_append = np.array(F_append, dtype=np.int32)
        F_prop_append = np.array(F_prop_append, dtype=np.int32)
        F = np.concatenate([F[idx_reamin, :], F_append])
        F_prop = np.concatenate([F_prop[idx_reamin], F_prop_append])
    OM_mesh = OM.TriMesh(V, F)
    FF = OM_mesh.face_face_indices()
    Q = queue.Queue()
    idx_face_edge = np.argwhere(F_prop >= 0)[:, 0].tolist()
    for idx in idx_face_edge:
        Q.put(idx)
    while not Q.empty():
        fid = Q.get()
        prop = F_prop[fid]
        near_fid = FF[fid, :]
        for i in range(3):
            fid_current = near_fid[i]
            if F_prop[fid_current] < 0:
                F_prop[fid_current] = prop
                Q.put(fid_current)
    for i in range(6):
        F_facei = F[np.argwhere(F_prop == i)[:, 0], :]
        v_facei = np.unique(F_facei.reshape([-1]))
        v_facei_T = np.zeros([V.shape[0]], dtype=bool)
        v_facei_T[v_facei] = True
        v_facei_T[V_prop >= 0] = False
        v_facei = np.argwhere(v_facei_T)[:, 0]
        V_prop[v_facei] = i

    facev_idx = np.argwhere(V_prop < 6)[:, 0]
    facev_faceid = V_prop[facev_idx]
    return V_mesh, facev_idx, facev_faceid, F, F_prop, edges_vertex_mesh_idx, edges_vertex_tar


def check_update(tm: TrainingManager):
    loss_terms = tm.loss_terms
    N_iter = len(loss_terms)
    weight = tm.weights
    best_idx = tm.best_idx
    loss_terms_best = loss_terms[best_idx]
    loss_terms_current = loss_terms[-1]
    loss_current = sum([loss_terms_current[i] * weight[i] for i in range(len(weight))])
    loss_best = sum([loss_terms_best[i] * weight[i] for i in range(len(weight))])
    status_record = tm.status
    bij_best = status_record[best_idx][0]
    bij_current = status_record[-1][0]
    if N_iter == 1:
        return True
    if bij_current > bij_best:
        return False
    if bij_current < bij_best:
        return True
    if loss_current < loss_best:
        return True
    return False


def cube_map_adjust(V, F, F_prop, edges_idx_list, edges_tar_list, method="rbf"):
    edges_idx, unique_idx = np.unique(np.concatenate(edges_idx_list), return_index=True)
    edges_tar = np.concatenate(edges_tar_list)[unique_idx, :]
    if method == "rbf":
        rbf = RBFInterpolator(V[edges_idx, :], edges_tar - V[edges_idx, :], kernel="linear")
        V_adjust = V + rbf(V)
        V_cen = V_adjust - 0.5
        scale = 0.5 / np.max(np.abs(V_cen), axis=1)
        V_adjust = V_cen * scale[:, None] + 0.5
    elif method == "optimize":
        device = torch.device("cuda")
        is_edge = np.zeros([V.shape[0]], dtype=bool)
        is_edge[edges_idx] = True
        is_inner = np.logical_not(is_edge)
        inner_idx = np.argwhere(is_inner)[:, 0]
        V_inner_prop = np.zeros([V.shape[0]], dtype=np.int32)
        V_inner_prop[F[:, 0]] = F_prop
        V_inner_prop[F[:, 1]] = F_prop
        V_inner_prop[F[:, 2]] = F_prop
        V_inner_prop = V_inner_prop[inner_idx]

        V_inner_prop_torch = torch.from_numpy(V_inner_prop.astype(np.int64)).to(device)

        FV = V[F, :]
        vec = FV[:, 1:, :] - FV[:, :1, :]
        vec_len = np.linalg.norm(vec, axis=2)
        vec2 = np.zeros_like(vec[:, :, :2])
        cos_vec = np.clip(np.sum(vec[:, 0, :] * vec[:, 1, :], axis=1) / (vec_len[:, 0] * vec_len[:, 1]), 1e-8, 1 - 1e-8)
        sin_vec = np.sqrt(1 - cos_vec ** 2)
        area = sin_vec * vec_len[:, 0] * vec_len[:, 1] / 2
        sin_vec = np.clip(sin_vec, a_min=1e-2, a_max=None)
        cos_vec = np.sign(cos_vec) * np.sqrt(1 - sin_vec ** 2)
        vec2[:, 0, 0] = np.linalg.norm(vec[:, 0, :], axis=1)
        vec2[:, 1, 0] = cos_vec * vec_len[:, 1]
        vec2[:, 1, 1] = sin_vec * vec_len[:, 1]
        vec2_inv = np.linalg.inv(vec2)
        vec2_inv_torch = torch.from_numpy(vec2_inv.astype(np.float32)).to(device)

        norm_tar = np.zeros(F.shape, dtype=np.float32)
        norm_tar[np.arange(F.shape[0]), F_prop % 3] = 2 * (F_prop // 3) - 1
        norm_tar_torch = torch.from_numpy(norm_tar.astype(np.float32)).to(device=device)
        NV_inner_arange_torch = torch.arange(inner_idx.shape[0], device=device)
        NF_arange_torch = torch.arange(F.shape[0], device=device)
        V_inner_torch = torch.from_numpy(V[inner_idx, :].astype(np.float32)).to(device=device)
        V_tar = V.copy()
        V_tar[edges_idx, :] = edges_tar
        V_adjust_torch = torch.from_numpy(V_tar.astype(np.float32)).to(device=device)
        inner_idx_torch = torch.from_numpy(inner_idx.astype(np.int64)).to(device)
        V_inner_adjust_torch = V_inner_torch.detach().clone()
        V_inner_adjust_torch = torch.nn.Parameter(V_inner_adjust_torch)
        F_torch = torch.from_numpy(F.astype(np.int64)).to(device)
        F_prop_torch = torch.from_numpy(F_prop.astype(np.int64)).to(device)
        area_torch = torch.from_numpy(area.astype(np.float32)).to(device)
        optim = torch.optim.Adam([V_inner_adjust_torch], lr=5e-4)
        tm = TrainingManager([V_inner_adjust_torch], loss_weights={"dis": 1e2, "err": 1e0, "norm": 1e2, "inv": 1e4, "arap": 10},
                             check_update=check_update, max_iter=300)
        for iter in range(300):
            with torch.no_grad():
                V_inner_tar_torch = V_inner_adjust_torch.detach().clone()
                V_inner_tar_torch[NV_inner_arange_torch, V_inner_prop_torch % 3] = (V_inner_prop_torch // 3) * 1.0
            loss_dis = torch.mean((V_inner_adjust_torch - V_inner_tar_torch) ** 2) * 3
            loss_err = torch.mean((V_inner_adjust_torch - V_inner_torch) ** 2) * 3
            V_adjust_torch = V_adjust_torch.detach()
            V_adjust_torch[inner_idx_torch, :] = V_inner_adjust_torch
            FV_adjust = V_adjust_torch[F_torch, :]
            vec_adjust = FV_adjust[:, 1:, :] - FV_adjust[:, :1, :]
            norm_adjust = torch.cross(vec_adjust[:, 0, :], vec_adjust[:, 1, :], dim=1)
            norm_adjust = norm_adjust / torch.clip(torch.norm(norm_adjust, dim=1)[:, None], 1e-6, None)
            loss_norm = torch.mean((norm_adjust - norm_tar_torch) ** 2) * 3
            loss_inv = torch.mean(penalty_pos(torch.sum(norm_adjust * norm_tar_torch, dim=1) - 0.5))
            n_inv = torch.sum(torch.sum(norm_adjust * norm_tar_torch, dim=1) < 0)
            F_axis = F_prop_torch % 3
            F_value = F_prop_torch // 3
            vec2_adjust = torch.zeros_like(vec_adjust[:, :, :2])
            vec2_adjust[NF_arange_torch, :, 1 - F_value] = vec_adjust[NF_arange_torch, :, (F_axis + 1) % 3]
            vec2_adjust[NF_arange_torch, :, F_value] = vec_adjust[NF_arange_torch, :, (F_axis + 2) % 3]
            J2T = vec2_inv_torch @ vec2_adjust
            singularvalues = torch.linalg.svdvals(J2T)
            loss_arap = torch.sum(area_torch[:, None] * (1 - singularvalues) ** 2)
            loss, _ = tm.record([loss_dis, loss_err, loss_norm, loss_inv, loss_arap], [n_inv], print_iter=20)
            optim.zero_grad()
            loss.backward()
            optim.step()
        V_inner_adjust_torch = tm.best_model[0]
        with torch.no_grad():
            V_inner_tar_torch = V_inner_adjust_torch.detach().clone()
            V_inner_tar_torch[NV_inner_arange_torch, V_inner_prop_torch % 3] = (V_inner_prop_torch // 3) * 1.0
            V_adjust_torch[inner_idx_torch, :] = V_inner_tar_torch
        V_adjust = V_adjust_torch.detach().cpu().numpy()

    return V_adjust


def remeshing(V, F):
    mesh_ml = ml.Mesh(V, F)
    ms = ml.MeshSet()
    ms.add_mesh(mesh_ml)
    # ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=2000)
    # ms.apply_filter("apply_coord_laplacian_smoothing", stepsmoothnum=n, boundary=True)
    ms.apply_filter('meshing_isotropic_explicit_remeshing')
    mesh_ml = ms.current_mesh()
    V2, F2 = mesh_ml.vertex_matrix(), mesh_ml.face_matrix()
    return V2, F2
