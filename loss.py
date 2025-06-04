import torch
import numpy as np
from utils import get_tar_cube_norm_torch, get_tar_cube_torch, get_tar_cube_star_torch, get_tar_cube_norm_star_torch, get_tar_semi_cube, penalty_pos

r = (3 / (4 * np.pi)) ** (1 / 3)


def loss_hessian(V, h):
    # V: N*19*3
    h2inv = 1.0 / h / h
    H_ii = (V[:, 1:4, :] + V[:, 4:7, :] - 2 * V[:, :1, :]) * h2inv
    H_ij = (V[:, 7:10, :] - V[:, 10:13, :] - V[:, 13:16, :] + V[:, 16:19, :]) * (h2inv / 4)
    loss_n = torch.sum(torch.sum(H_ii ** 2, dim=1) + torch.sum(H_ij ** 2, dim=1) * 2, dim=1)
    loss = torch.mean(loss_n)
    return loss


def loss_inner(J, arap_ratio=1, bij_v=0, inv=True, align=False):
    detJ = torch.det(J)
    pos_detJ = torch.sign(detJ).detach()
    bij = torch.sum(detJ < 0).item()
    if inv:
        weight = torch.abs(detJ).detach()
    else:
        weight = torch.ones_like(detJ)
    singular_values = torch.linalg.svdvals(J)
    singular_values = singular_values * pos_detJ.reshape([-1, 1])
    # singular_values = fast_svdvals(J)
    if inv:
        singular_values = 1 / singular_values
    loss_inv = torch.mean(penalty_pos(detJ - bij_v) ** 2)
    loss_arap = (singular_values - 1) ** 2 * weight.reshape([-1, 1])
    if arap_ratio == 1:
        loss_arap = torch.mean(loss_arap)
    else:
        singular_values_ave = torch.mean(singular_values, dim=1).detach()
        loss_asap = (singular_values - singular_values_ave.reshape([-1, 1])) ** 2 * weight.reshape([-1, 1])
        loss_arap = torch.mean(loss_asap * (1 - arap_ratio) + arap_ratio * loss_arap)
    if align:
        J_inv = torch.inverse(J)
        J_inv_n = J_inv / torch.norm(J_inv, dim=1).unsqueeze(1)
        dx, dy, dz = J_inv_n[:, :, 0], J_inv_n[:, :, 1], J_inv_n[:, :, 2]
        loss_align = torch.mean((torch.sum(dx * dy, dim=1) ** 2 + torch.sum(dy * dz, dim=1) ** 2 + torch.sum(dx * dz, dim=1) ** 2) * weight)
        return loss_align, loss_inv, loss_arap, bij
    return loss_inv, loss_arap, bij


def loss_bound_cube(bound, faces, tar=None, tar_method="mindis"):
    # loss_haus : d(p,\partial [0,1]^3)
    if tar_method == "mindis":
        tar_func = get_tar_cube_torch
        tar_norm_func = get_tar_cube_norm_torch
    elif tar_method == "star":
        tar_func = get_tar_cube_star_torch
        tar_norm_func = get_tar_cube_norm_star_torch
    if tar is None:
        tar = tar_func(bound)
    err = tar - bound
    dist2 = torch.sum(err ** 2, dim=1)
    loss_dis = torch.mean(dist2)
    triangles = bound[faces, :]
    norm = (torch.cross(triangles[:, 1, :] - triangles[:, 0, :], triangles[:, 2, :] - triangles[:, 0, :], dim=1))
    norm_len = torch.norm(norm, dim=1).reshape([-1, 1])
    norm = norm / norm_len
    # center = torch.mean(triangles, dim=1)
    triangles_tar = tar[faces, :]
    norm_tar = (torch.cross(triangles_tar[:, 1, :] - triangles_tar[:, 0, :], triangles_tar[:, 2, :] - triangles_tar[:, 0, :], dim=1))
    norm_tar_len = torch.clip(torch.norm(norm_tar, dim=1).reshape([-1, 1]).detach(), 1e-5, None)
    norm_tar = norm_tar / norm_tar_len
    face_cen = torch.mean(triangles, dim=1)
    norm_tar_cen = tar_norm_func(face_cen)
    norm_tar = torch.sign(torch.sum(norm_tar * norm_tar_cen, dim=1))[:, None] * norm_tar
    err = norm - norm_tar
    dist2 = torch.sum(err ** 2, dim=1)
    loss_norm_pos = torch.mean(penalty_pos(torch.sum(norm * norm_tar, dim=1)))
    loss_norm = torch.mean(dist2) + loss_norm_pos * 100
    return loss_dis, loss_norm


def loss_bound_mesh(bound, faces, tar):
    # loss_haus : d(p,\partial [0,1]^3)
    err = tar - bound
    dist2 = torch.sum(err ** 2, dim=1)
    loss_dis = torch.mean(dist2)
    triangles = bound[faces, :]
    norm = (torch.cross(triangles[:, 1, :] - triangles[:, 0, :], triangles[:, 2, :] - triangles[:, 0, :], dim=1))
    norm_len = torch.clip(torch.norm(norm, dim=1).reshape([-1, 1]), min=1e-6)
    norm = norm / norm_len
    # center = torch.mean(triangles, dim=1)
    triangles_tar = tar[faces, :]
    norm_tar = (torch.cross(triangles_tar[:, 1, :] - triangles_tar[:, 0, :], triangles_tar[:, 2, :] - triangles_tar[:, 0, :], dim=1))
    norm_tar_len = torch.clip(torch.norm(norm_tar, dim=1).reshape([-1, 1]).detach(), 1e-5, None)
    norm_tar = norm_tar / norm_tar_len
    err = norm - norm_tar
    dist2 = torch.sum(err ** 2, dim=1)
    # loss_norm_pos = torch.mean(penalty_pos(torch.sum(norm * norm_tar, dim=1)))
    loss_norm = torch.mean(dist2)  # + loss_norm_pos * 100
    return loss_dis, loss_norm


def loss_bound_cube_cut(out_bound, corner_idx, tar_corners, edges_idx_cat, tar_edges, vec_edge_tar_n, Index_edges,
                        faces, faces_faceid, facev_idx, facev_faceid):
    out_corners = out_bound[corner_idx, :]
    loss_corners = torch.mean(torch.sum((out_corners - tar_corners) ** 2, dim=-1))
    out_edges_cat = out_bound[edges_idx_cat, :]
    # with torch.no_grad():
    #     err_edge = out_edges_cat - tar_edges
    #     temp = torch.sum(err_edge * vec_edge_tar_n, dim=1)
    #     tar_edges = tar_edges + temp[:, None] * vec_edge_tar_n
    loss_edge_dis = torch.mean(torch.sum((out_edges_cat - tar_edges) ** 2, dim=1))

    vec_edge = out_edges_cat[1:, :] - out_edges_cat[:-1, :]
    vec_edge[Index_edges[1:-1] - 1, :] = 1
    vec_edge_n = vec_edge / torch.norm(vec_edge, dim=1)[:, None]
    err_vec_edge = torch.sum((vec_edge_n - vec_edge_tar_n[:-1, :]) ** 2, dim=1)
    err_vec_edge[Index_edges[1:-1] - 1] = 0
    err_vec_edge_cumsum = torch.cat([torch.zeros([1], dtype=out_bound.dtype, device=out_bound.device), torch.cumsum(err_vec_edge, dim=0)])
    err_vec_edges_mean = (err_vec_edge_cumsum[Index_edges[1:] - 1] - err_vec_edge_cumsum[Index_edges[:-1]]) / (Index_edges[1:] - Index_edges[:-1] - 1)
    loss_edge_vec = torch.mean(err_vec_edges_mean)

    out_facev = out_bound[facev_idx, :]
    Nv = facev_idx.shape[0]
    Nf = faces.shape[0]
    tar_norm = torch.zeros([Nf, 3], dtype=out_bound.dtype, device=out_bound.device)
    with torch.no_grad():
        tar_facev = torch.clip(out_facev.detach(), 0, 1)
        tar_facev[torch.arange(Nv), facev_faceid % 3] = 1.0 * (facev_faceid // 3)
        tar_norm[torch.arange(Nf), faces_faceid % 3] = 2.0 * (faces_faceid // 3) - 1
    loss_dis = torch.mean(torch.sum((out_facev - tar_facev) ** 2, dim=-1))
    triangles = out_bound[faces, :]
    norm = (torch.cross(triangles[:, 1, :] - triangles[:, 0, :], triangles[:, 2, :] - triangles[:, 0, :], dim=1))
    norm_len = torch.norm(norm, dim=1).reshape([-1, 1])
    norm = norm / norm_len
    loss_norm = torch.mean(torch.sum((norm - tar_norm) ** 2, dim=-1))
    return loss_corners, loss_edge_dis, loss_edge_vec, loss_dis, loss_norm


def loss_bound_semi_cube(bound, faces, d=0.5):
    # r = (3/(4pi))^(1/3)
    triangles = bound[faces, :]
    norm = (torch.cross(triangles[:, 1, :] - triangles[:, 0, :], triangles[:, 2, :] - triangles[:, 0, :], dim=1))
    norm_len = torch.norm(norm, dim=1).reshape([-1, 1])
    norm = norm / norm_len
    tar, norm_tar = get_tar_semi_cube(bound, d)
    norm_tar = torch.mean(norm_tar[faces, :], dim=1)
    err = tar - bound
    dist2 = torch.sum(err ** 2, dim=1)
    loss_dis = torch.mean(dist2)
    err_norm = norm - norm_tar
    dist2_norm = torch.sum(err_norm ** 2, dim=1)
    loss_norm = torch.mean(dist2_norm)
    return loss_dis, loss_norm


def loss_star_shape(bound, faces, center=0.5):
    triangles = bound[faces, :]
    norm = (torch.cross(triangles[:, 1, :] - triangles[:, 0, :], triangles[:, 2, :] - triangles[:, 0, :], dim=1))
    norm_len = torch.norm(norm, dim=1).reshape([-1, 1])
    norm = norm / norm_len
    center = torch.mean(triangles - center, dim=1)
    center_len = torch.norm(center, dim=1).reshape([-1, 1])
    center = center / center_len
    loss = penalty_pos(torch.sum(center * norm, dim=1))
    return torch.mean(loss ** 2)
