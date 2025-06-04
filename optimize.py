import colorcet
import trimesh
import BSplineVolumn
import utils
from loss import *
from utils import draw_tri_mesh
from PIL import Image
import pyvista as pv
from training_manager import TrainingManager


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


def optimize_semi_cube(id, pl: pv.Plotter, config, netdp, bound, inner, corner_idx=None):
    device = config.device
    lr = config.lr[id]
    max_iter = config.iteration[id]
    bij_v = config.bij_v
    weight = config.weight[id]
    min_iter = config.min_iteration[id]
    stop_dis = config.stop_dis[id]
    optim = torch.optim.Adam(netdp.parameters(), lr=lr)
    points, cubes, hes, h = inner
    points = np.copy(points).astype(np.float32)
    points_bound = np.copy(bound.vertices).astype(np.float32)
    points_torch = torch.from_numpy(points).to(device)
    points_bound_torch = torch.from_numpy(points_bound).to(device)
    faces_torch = torch.from_numpy(bound.faces.astype(np.int64)).to(device)
    cubes_torch = torch.from_numpy(cubes.astype(np.int64)).to(device)
    hes_torch = torch.from_numpy(hes.astype(np.int64)).to(device)
    if corner_idx is not None:
        corner_idx_torch = torch.from_numpy(corner_idx.astype(np.int64)).to(device)
        corners_cube = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
        corners_cube_torch = torch.from_numpy(corners_cube).to(device)
        corners_1_torch = utils.get_tar_semi_cube(corners_cube_torch, 1)[0]
        corners_2_torch = utils.get_tar_semi_cube(corners_cube_torch, 2)[0]
    else:
        corner_idx_torch = torch.zeros([0], dtype=torch.int64).to(device)
        corners_1_torch = torch.zeros([0, 3], dtype=torch.float32).to(device)
        corners_2_torch = torch.zeros([0, 3], dtype=torch.float32).to(device)
    N_inner = points.shape[0]
    tm = TrainingManager(netdp, loss_weights=weight, check_update=check_update, max_iter=max_iter)
    utils.pv_reset_plotter(pl)
    pl.render()
    pl_mesh = None

    for iter in range(max_iter):
        if iter < 600:
            d = 2
            corners_torch = corners_2_torch
        else:
            d = 1
            corners_torch = corners_1_torch
        movement = (torch.rand([1, 3], dtype=torch.float32, device=device) - 0.5) * h
        points_torch_move = points_torch + movement
        all_input = torch.cat([points_torch_move, points_bound_torch])
        all_output = netdp(all_input)
        out_inner = all_output[:N_inner, :]
        out_bound = all_output[N_inner:, :]
        loss_corner = torch.sum((out_bound[corner_idx_torch, :] - corners_torch) ** 2) / (max(1, corner_idx_torch.shape[0]))
        # N*19*3
        hes_vertices = out_inner[hes_torch, :]
        # cube_vertices = out_inner[cubes_torch, :]
        loss_hes = loss_hessian(hes_vertices, h)
        # N,(3,3),3
        # J_inner = ((cube_vertices[:, [1, 3, 4], :] - cube_vertices[:, [0], :]) / h).permute([0, 2, 1])
        J_inner = ((hes_vertices[:, 1:4, :] - hes_vertices[:, 4:7, :]) / (2 * h)).permute([0, 2, 1])
        loss_bij, loss_arap, bij = loss_inner(J_inner, bij_v=bij_v)
        loss_dis, loss_norm = loss_bound_semi_cube(out_bound, faces_torch, d=d)
        loss_star = loss_star_shape(out_bound, faces_torch)
        loss, print_info = tm.record([loss_norm, loss_dis, loss_bij, loss_star, loss_arap, loss_hes, loss_corner], [bij], print_iter=100)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if print_info and config.plot:
            plot_vertices = out_bound.detach().cpu().numpy()
            plot_faces = bound.faces
            plot_bound = trimesh.Trimesh(plot_vertices, plot_faces)
            pl_mesh = draw_tri_mesh(pl, plot_bound, cube=config.draw_cube, update=pl_mesh)
            pl.render()
            if config.save_fig:
                path = "./img/%s_step1_%5d.png" % (config.name, iter)
                Image.fromarray(np.array(pl.screenshot(filename=None, transparent_background=True, return_img=True))).save(path)
        if tm.loss_terms[tm.best_idx][1] < stop_dis and iter > min_iter:
            break
    netdp.load_state_dict(tm.best_model)
    return tm.best_model


def optimize_cube(id, pl, config, netdp, bound, inner, corner_idx=None):
    device = config.device
    lr = config.lr[id]
    max_iter = config.iteration[id]
    min_iter = config.min_iteration[id]
    stop_dis = config.stop_dis[id]
    bij_v = config.bij_v
    weight = config.weight[id]
    optim = torch.optim.Adam(netdp.parameters(), lr=lr)
    points, cubes, hes, h = inner
    points = np.copy(points).astype(np.float32)
    points_bound = np.copy(bound.vertices).astype(np.float32)
    faces = np.copy(bound.faces).astype(np.int64)
    points_bound_torch = torch.from_numpy(points_bound).to(device)
    points_torch = torch.from_numpy(points).to(device)
    faces_torch = torch.from_numpy(faces).to(device)
    cubes_torch = torch.from_numpy(cubes.astype(np.int64)).to(device)
    hes_torch = torch.from_numpy(hes.astype(np.int64)).to(device)
    if corner_idx is not None:
        corner_idx_torch = torch.from_numpy(corner_idx.astype(np.int64)).to(device)
        corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
        corners_torch = torch.from_numpy(corners).to(device)
    else:
        corner_idx_torch = torch.zeros([0], dtype=torch.int64).to(device)
        corners_torch = torch.zeros([0, 3], dtype=torch.float32).to(device)
    N_inner = points.shape[0]
    # best = {"loss": None, "dis": None, "norm": None, "bij": False, "arap": None, "align": None, "bound_arap": None, "param": None, "print": None}
    tm = TrainingManager(netdp, loss_weights=weight, check_update=check_update, max_iter=max_iter)
    utils.pv_reset_plotter(pl)
    pl_mesh = None
    for iter in range(max_iter):
        movement = (torch.rand([1, 3], dtype=torch.float32, device=device) - 0.5) * h
        points_torch_move = points_torch + movement
        all_input = torch.cat([points_torch_move, points_bound_torch])
        all_output = netdp(all_input)
        out_inner = all_output[:N_inner, :]
        out_bound = all_output[N_inner:, :]
        loss_corner = torch.sum((out_bound[corner_idx_torch, :] - corners_torch) ** 2) / (max(1, corner_idx_torch.shape[0]))
        # N*19*3
        hes_vertices = out_inner[hes_torch, :]
        # cube_vertices = out_inner[cubes_torch]
        loss_hes = loss_hessian(hes_vertices, h)
        # N,(3,3),3
        # J_inner = ((cube_vertices[:, [1, 3, 4], :] - cube_vertices[:, [0], :]) / h).permute([0, 2, 1])
        J_inner = ((hes_vertices[:, 1:4, :] - hes_vertices[:, 4:7, :]) / (2 * h)).permute([0, 2, 1])
        loss_bij, loss_arap, bij = loss_inner(J_inner, bij_v=bij_v)
        # loss_dis, loss_norm = loss_bound_semi_cube(out_bound, faces_torch, min_dis=min_dis,d=0.05)
        loss_dis, loss_norm = loss_bound_cube(out_bound, faces_torch)
        loss_star = loss_star_shape(out_bound, faces_torch)
        loss, print_info = tm.record([loss_norm, loss_dis, loss_bij, loss_star, loss_arap, loss_hes, loss_corner], [bij], print_iter=100)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if print_info and config.plot:
            plot_vertices = out_bound.detach().cpu().numpy()
            plot_faces = bound.faces
            plot_bound = trimesh.Trimesh(plot_vertices, plot_faces)
            pl_mesh = draw_tri_mesh(pl, plot_bound, cube=config.draw_cube, update=pl_mesh)
            pl.render()
            if config.save_fig:
                path = "./img/%s_step2_%5d.png" % (config.name, iter)
                Image.fromarray(np.array(pl.screenshot(filename=None, transparent_background=True, return_img=True))).save(path)
        if tm.loss_terms[tm.best_idx][1] < stop_dis and iter > min_iter:
            break
    netdp.load_state_dict(tm.best_model)
    return tm.best_model


def optimize_cube_fixcorner(id, pl, config, netdp, bound, inner, corner_idx):
    device = config.device
    lr = config.lr[id]
    max_iter = config.iteration[id]
    min_iter = config.min_iteration[id]
    stop_dis = config.stop_dis[id]
    bij_v = config.bij_v
    weight = config.weight[id]
    optim = torch.optim.Adam(netdp.parameters(), lr=lr)
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
    corners_torch = torch.from_numpy(corners).to(device)
    corner_idx_torch = torch.from_numpy(corner_idx.astype(np.int64)).to(device)
    points, cubes, hes, h = inner
    points = np.copy(points).astype(np.float32)
    points_bound = np.copy(bound.vertices).astype(np.float32)
    faces = np.copy(bound.faces).astype(np.int64)
    points_bound_torch = torch.from_numpy(points_bound).to(device)
    points_torch = torch.from_numpy(points).to(device)
    faces_torch = torch.from_numpy(faces).to(device)
    cubes_torch = torch.from_numpy(cubes.astype(np.int64)).to(device)
    hes_torch = torch.from_numpy(hes.astype(np.int64)).to(device)
    N_inner = points.shape[0]
    # best = {"loss": None, "dis": None, "norm": None, "bij": False, "arap": None, "align": None, "bound_arap": None, "param": None, "print": None}
    tm = TrainingManager(netdp, loss_weights=weight, check_update=check_update, max_iter=max_iter)
    utils.pv_reset_plotter(pl)
    pl_mesh = None
    pl_corner = None
    for iter in range(max_iter):
        movement = (torch.rand([1, 3], dtype=torch.float32, device=device) - 0.5) * h
        points_torch_move = points_torch + movement
        all_input = torch.cat([points_torch_move, points_bound_torch])
        all_output = netdp(all_input)
        out_inner = all_output[:N_inner, :]
        out_bound = all_output[N_inner:, :]
        # N*19*3
        hes_vertices = out_inner[hes_torch, :]
        # cube_vertices = out_inner[cubes_torch, :]
        loss_hes = loss_hessian(hes_vertices, h)
        # N,(3,3),3
        # J_inner = ((cube_vertices[:, [1, 3, 4], :] - cube_vertices[:, [0], :]) / h).permute([0, 2, 1])
        J_inner = ((hes_vertices[:, 1:4, :] - hes_vertices[:, 4:7, :]) / (2 * h)).permute([0, 2, 1])
        loss_bij, loss_arap, bij = loss_inner(J_inner, bij_v=bij_v)
        loss_dis, loss_norm = loss_bound_cube(out_bound, faces_torch)
        loss_corner = torch.mean((out_bound[corner_idx_torch, :] - corners_torch) ** 2)
        loss_star = loss_star_shape(out_bound, faces_torch)
        loss, print_info = tm.record([loss_norm, loss_dis, loss_bij, loss_star, loss_arap, loss_hes, loss_corner], [bij], print_iter=100)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if print_info and config.plot:
            plot_vertices = out_bound.detach().cpu().numpy()
            plot_faces = bound.faces
            plot_bound = trimesh.Trimesh(plot_vertices, plot_faces)
            pl_mesh = draw_tri_mesh(pl, plot_bound, cube=config.draw_cube, update=pl_mesh)
            if pl_corner is None:
                pl_corner = pv.PolyData(plot_vertices[corner_idx, :])
                pl.add_mesh(pl_corner, point_size=10, render_points_as_spheres=True, color="g", name="corner")
            else:
                pl_corner.points = plot_vertices[corner_idx, :]
            pl.render()
            if config.save_fig:
                path = "./img/%s_step3_%5d.png" % (config.name, iter)
                Image.fromarray(np.array(pl.screenshot(filename=None, transparent_background=True, return_img=True))).save(path)
        if tm.loss_terms[tm.best_idx][1] < stop_dis and iter > min_iter:
            break
    netdp.load_state_dict(tm.best_model)
    return tm.best_model


def optimize_cube_fixedge_cut(id, pl, config, netdp, bound, inner, corner_idx, edge_info, facev_info, F_faceid):
    edges_idx_cat, tar_edges_cat, Index_edges_cat = edge_info
    facev_idx, facev_faceid = facev_info
    device = config.device
    lr = config.lr[id]
    max_iter = config.iteration[id]
    min_iter = config.min_iteration[id]
    stop_dis = config.stop_dis[id]
    bij_v = config.bij_v
    weight = config.weight[id]
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
    optim = torch.optim.Adam(netdp.parameters(), lr=lr)
    points, cubes, hes, h = inner
    points = np.copy(points).astype(np.float32)
    points_bound = np.copy(bound.vertices).astype(np.float32)
    edges_idx_cat = np.copy(edges_idx_cat.astype(np.int64))
    tar_edges_cat = np.copy(tar_edges_cat.astype(np.float32))
    Index_edges_cat = np.copy(Index_edges_cat.astype(np.int64))
    facev_idx = np.copy(facev_idx.astype(np.int64))
    facev_faceid = np.copy(facev_faceid.astype(np.int64))
    F_faceid = np.copy(F_faceid.astype(np.int64))
    N_bound = points_bound.shape[0]
    N_inner = points.shape[0]

    points_torch = torch.from_numpy(points).to(device)
    points_bound_torch = torch.from_numpy(points_bound).to(device)
    corners_torch = torch.from_numpy(corners).to(device)
    corner_idx_torch = torch.from_numpy(corner_idx.astype(np.int64)).to(device)
    edges_idx_cat_torch = torch.from_numpy(edges_idx_cat).to(device)
    tar_edges_cat_torch = torch.from_numpy(tar_edges_cat).to(device)
    Index_edges_cat_torch = torch.from_numpy(Index_edges_cat).to(device)
    facev_idx_torch = torch.from_numpy(facev_idx).to(device)
    facev_faceid_torch = torch.from_numpy(facev_faceid).to(device)
    F_faceid_torch = torch.from_numpy(F_faceid).to(device)
    faces_torch = torch.from_numpy(bound.faces.astype(np.int64)).to(device)
    cubes_torch = torch.from_numpy(cubes.astype(np.int64)).to(device)
    hes_torch = torch.from_numpy(hes.astype(np.int64)).to(device)
    tm = TrainingManager(netdp, loss_weights=weight, check_update=check_update, max_iter=max_iter)
    utils.pv_reset_plotter(pl)
    pl_mesh = None
    pl_corner = None
    Np_edges = Index_edges_cat_torch[1:] - Index_edges_cat_torch[:-1]
    vec_cube_edge = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]],
        dtype=torch.float32, device=device)
    vec_edge_tar_n = torch.repeat_interleave(vec_cube_edge, repeats=Np_edges, dim=0)
    for iter in range(max_iter):
        movement = (torch.rand([1, 3], dtype=torch.float32, device=device) - 0.5) * h
        points_torch_move = points_torch + movement
        all_input = torch.cat([points_torch_move, points_bound_torch])
        all_output = netdp(all_input)
        out_inner = all_output[:N_inner, :]
        out_bound = all_output[N_inner:N_inner + N_bound, :]
        # N*19*3
        hes_vertices = out_inner[hes_torch, :]
        # cube_vertices = out_inner[cubes_torch, :]
        loss_hes = loss_hessian(hes_vertices, h)
        # N,(3,3),3
        # J_inner = ((cube_vertices[:, [1, 3, 4], :] - cube_vertices[:, [0], :]) / h).permute([0, 2, 1])
        J_inner = ((hes_vertices[:, 1:4, :] - hes_vertices[:, 4:7, :]) / (2 * h)).permute([0, 2, 1])
        loss_bij, loss_arap, bij = loss_inner(J_inner, bij_v=bij_v)
        loss_corners, loss_edge, loss_edge_vec, loss_dis, loss_norm = \
            loss_bound_cube_cut(out_bound, corner_idx_torch, corners_torch, edges_idx_cat_torch, tar_edges_cat_torch, vec_edge_tar_n,
                                Index_edges_cat_torch,
                                faces_torch, F_faceid_torch, facev_idx_torch, facev_faceid_torch)
        loss_star = loss_star_shape(out_bound, faces_torch)
        out_bound_clip = torch.clip(out_bound, 0 - 1e-4, 1 + 1e-4).detach()
        loss_out = torch.mean((out_bound - out_bound_clip) ** 2) * 3
        loss, print_info = tm.record([loss_norm, loss_dis, loss_out, loss_bij, loss_star, loss_arap, loss_hes,
                                      loss_corners, loss_edge, loss_edge_vec], [bij], print_iter=100)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if print_info and config.plot:
            plot_vertices = out_bound.detach().cpu().numpy()
            plot_faces = bound.faces
            plot_bound = trimesh.Trimesh(plot_vertices, plot_faces)
            pl_mesh = draw_tri_mesh(pl, plot_bound, cube=config.draw_cube, update=pl_mesh)
            if pl_corner is None:
                pl_corner = pv.PolyData(plot_vertices[corner_idx, :])
                pl.add_mesh(pl_corner, point_size=10, render_points_as_spheres=True, color="g", name="corner")
            else:
                pl_corner.points = plot_vertices[corner_idx, :]
            pl.render()
            if config.save_fig:
                path = "./img/%s_step4_%5d.png" % (config.name, iter)
                Image.fromarray(np.array(pl.screenshot(filename=None, transparent_background=True, return_img=True))).save(path)
        if tm.loss_terms[tm.best_idx][1] < stop_dis and iter > min_iter:
            break
    netdp.load_state_dict(tm.best_model)
    return tm.best_model


def optimize_bs(id, pl, config, bs: BSplineVolumn.BS_Torch, V_uvw, V_tar, F):
    device = config.device
    lr = config.lr[id]
    max_iter = config.iteration[id]
    min_iter = config.min_iteration[id]
    stop_dis = config.stop_dis[id]
    bij_v = 0.2
    weight = config.weight[id]

    optim = torch.optim.Adam(bs.parameters(), lr=lr)

    tm = TrainingManager(bs, loss_weights=weight, check_update=check_update, max_iter=max_iter)

    cube_norm = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32)

    N_sample = 101
    # points_uvw = torch.from_numpy(utils.get_uvw(N_sample).astype(np.float32)).to(device)
    points_uvw = torch.zeros([N_sample, N_sample, N_sample, 3], device=device, dtype=torch.float32)
    V_uvw_torch = torch.from_numpy(V_uvw.astype(np.float32)).to(device)
    V_tar_torch = torch.from_numpy(V_tar.astype(np.float32)).to(device)
    F_torch = torch.from_numpy(F.astype(np.int64)).to(device)
    utils.pv_reset_plotter(pl)
    pl_mesh = None
    tri_v, tri = utils.get_surface_triangles(N_sample, N_sample, N_sample)

    for iter in range(max_iter):
        if iter == 0:
            tm.weights[0] /= 500
        if iter == 20:
            tm.weights[0] *= 10
        if iter == 40:
            tm.weights[0] *= 10
        if iter == 80:
            tm.weights[0] *= 5
        V_xyz = bs(V_uvw_torch)
        loss_bound_dis, loss_bound_norm = loss_bound_mesh(V_xyz, F_torch, V_tar_torch)
        len_uvw = torch.rand([3, N_sample], device=device, dtype=torch.float32) + 0.1
        len_uvw[:, 0] = 0
        len_uvw /= len_uvw.sum(dim=1)[:, None]
        param_uvw = torch.cumsum(len_uvw, dim=1)
        points_uvw[:, :, :, 0] = param_uvw[0, :, None, None]
        points_uvw[:, :, :, 1] = param_uvw[1, None, :, None]
        points_uvw[:, :, :, 2] = param_uvw[2, None, None, :]
        points_xyz = bs(points_uvw)

        Du2 = (points_xyz[1:, :, :, :] - points_xyz[:-1, :, :, :]) / len_uvw[0, 1:, None, None, None]
        Dv2 = (points_xyz[:, 1:, :, :] - points_xyz[:, :-1, :, :]) / len_uvw[1, None, 1:, None, None]
        Dw2 = (points_xyz[:, :, 1:, :] - points_xyz[:, :, :-1, :]) / len_uvw[2, None, None, 1:, None]
        llu = (len_uvw[0, 1:-1, None, None, None] + len_uvw[0, 2:, None, None, None]) / 2
        llv = (len_uvw[1, None, 1:-1, None, None] + len_uvw[1, None, 2:, None, None]) / 2
        llw = (len_uvw[2, None, None, 1:-1, None] + len_uvw[2, None, None, 2:, None]) / 2

        Duu2 = (Du2[1:, :, :, :] - Du2[:-1, :, :, :]) / llu
        Dvv2 = (Dv2[:, 1:, :, :] - Dv2[:, :-1, :, :]) / llv
        Dww2 = (Dw2[:, :, 1:, :] - Dw2[:, :, :-1, :]) / llw
        Duv2 = (Du2[:, 1:, :, :] - Du2[:, :-1, :, :]) / len_uvw[1, None, 1:, None, None]
        Duw2 = (Du2[:, :, 1:, :] - Du2[:, :, :-1, :]) / len_uvw[2, None, None, 1:, None]
        Dvw2 = (Dv2[:, :, 1:, :] - Dv2[:, :, :-1, :]) / len_uvw[2, None, None, 1:, None]
        loss_smooth2 = (torch.mean(Duu2 ** 2) + torch.mean(Dvv2 ** 2) + torch.mean(Dww2 ** 2)) * 3 + \
                       (torch.mean(Duv2 ** 2) + torch.mean(Duw2 ** 2) + torch.mean(Dvw2 ** 2)) * 6
        Du2_pad = torch.cat([Du2, Du2[-1:, :, :, :]], dim=0)
        Dv2_pad = torch.cat([Dv2, Dv2[:, -1:, :, :]], dim=1)
        Dw2_pad = torch.cat([Dw2, Dw2[:, :, -1:, :]], dim=2)
        J2 = torch.stack([Du2_pad, Dv2_pad, Dw2_pad], dim=4)
        detJ2 = torch.det(J2)
        singular_values = torch.linalg.svdvals(J2)
        singular_values = singular_values * torch.sign(detJ2).detach()[:, :, :, None]
        loss_arap = torch.mean(torch.sum((singular_values - 1) ** 2, dim=-1))
        bij = torch.sum(detJ2 < 0.0).item()
        loss_inv = torch.mean(penalty_pos(detJ2 - bij_v) ** 2)

        loss, print_info = tm.record([loss_inv, loss_arap, loss_smooth2, loss_bound_dis, loss_bound_norm], [bij],
                                     print_iter=20)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if print_info and config.plot:
            map_xyz = points_xyz.detach().cpu().numpy()
            tri_v_coord = map_xyz[tri_v[:, 0], tri_v[:, 1], tri_v[:, 2]]
            if pl_mesh is None:
                pl_mesh = [pv.make_tri_mesh(tri_v_coord, tri[i]) for i in range(6)]
                for i in range(6):
                    pl.add_mesh(pl_mesh[i], color=colorcet.glasbey_light[i], name=str(i))
                pl.reset_camera()
            else:
                for i in range(6):
                    pl_mesh[i].points = tri_v_coord
            pl.render()
            if config.save_fig:
                path = "./img/%s_step5_%5d.png" % (config.name, iter)
                Image.fromarray(np.array(pl.screenshot(filename=None, transparent_background=True, return_img=True))).save(path)
        if tm.loss_terms[tm.best_idx][0] < stop_dis and iter > min_iter:
            break
    bs.load_state_dict(tm.best_model)
    return tm.best_model
