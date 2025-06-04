import os
import joblib
import pyvista as pv
from net import Net
from optimize import *
from sampling import InteriorSample
import time
import trimesh
from PIL import Image
from BSplineVolumn import BS_Torch
from scipy.interpolate import RBFInterpolator


class Config():
    def __init__(self):
        self.device = torch.device("cuda")
        self.N_sample = 41
        self.N_fitting = 41
        self.N_coeff = 30
        self.order = 4
        self.bij_v = 1e-1
        self.lr = [5e-4, 5e-4, 5e-4, 5e-4, 5e-4]
        self.iteration = [1000, 1000, 500, 1000, 100]
        self.min_dis = [0, 0, 0, 0, 0, 0, 0]
        self.stop_dis = [0, 0, 0, 0, 0, 0, 0]
        self.min_iteration = [1000, 1000, 500, 1000, 40]
        self.weight = [{"norm": 1e1, "dis": 1e3, "bij": 1e6, "star": 1e3, "arap": 1, "fair": 0.002, "corner": 0},
                       {"norm": 1e1, "dis": 1e3, "bij": 1e6, "star": 1e3, "arap": 1, "fair": 0.002, "corner": 0},
                       {"norm": 1e1, "dis": 1e4, "bij": 1e6, "star": 1e3, "arap": 1, "fair": 0.002, "corner": 1e3},
                       {"norm": 1e1, "dis": 1e4, "out": 0, "bij": 1e6, "star": 1e3, "arap": 1, "fair": 0.002, "corner": 1e3, "edge": 1e4,
                        "edge_v": 1e1},
                       {"bij": 5e4, "arap": 1e-3, "fair": 5e-3, "bound_dis": 5e4, "bound_norm": 2e1},
                       ]
        self.name = None
        self.plot = True
        self.save_fig = True
        self.source_dir = "./mesh"
        self.save_dir = "./result"
        self.img_dir = "./img"
        self.seed = 765
        self.draw_cube = False


def savegraphic(pl, path):
    while os.path.exists(path):
        num = int(path[-7:-4]) + 1
        path = path[:-7] + "%03d" % num + path[-4:]
    Image.fromarray(np.array(pl.screenshot(filename=None, transparent_background=True, return_img=True))).save(path)


def run(config):
    T_start = time.time()
    # Data
    source_dir = config.source_dir
    save_dir = config.save_dir
    model_name = config.name
    mesh_filename = model_name + ".obj"
    seed = config.seed

    utils.setup_seed(seed)
    mesh_path = os.path.join(source_dir, mesh_filename)
    bound = trimesh.load_mesh(mesh_path)
    V, F = bound.vertices.astype(np.float32), bound.faces.astype(np.int32)
    V2, F2 = utils.remeshing(V, F)
    bound_remesh = trimesh.Trimesh(V2, F2)
    bound_remesh.export(os.path.join(save_dir, model_name + "_remeshing.obj"))
    cover_vertices, cover_cubes, cover_hes, h = InteriorSample(V2, F2, N=10000, expand=2)
    inner = [cover_vertices.astype(np.float32), cover_cubes.astype(np.int64), cover_hes.astype(np.int64), h]
    volumn = bound_remesh.volume
    scale = np.power(1 / volumn, 1 / 3)
    surface_sample = V2[np.random.choice(V2.shape[0], 1000), :]
    inner_sample = cover_vertices[np.random.choice(cover_vertices.shape[0], 1000), :]
    distance = np.min(np.linalg.norm(inner_sample.reshape([1, -1, 3]) - surface_sample.reshape([-1, 1, 3]), axis=2), axis=0)
    center_idx = np.argmax(distance)
    center_point = inner_sample[center_idx, :]
    V3 = (V2 - center_point) * scale + 0.5
    bound_remesh_scale = trimesh.Trimesh(V3, F2)
    bound_remesh_scale.export(os.path.join(save_dir, model_name + "_remeshing.obj"))
    inner_scale = [inner[0].copy(), inner[1].copy(), inner[2].copy(), inner[3]]
    inner_scale[0] = ((cover_vertices - center_point) * scale + 0.5).astype(np.float32)
    inner_scale[3] *= scale

    device = config.device
    net = Net().to(device)
    N_coeff = config.N_coeff
    bs_forward = BS_Torch(N_coeff, N_coeff, N_coeff, config.order).to(device)
    try:
        net.load_state_dict(torch.load("identity.pth"))
    except:
        net.initialize()
        net.load_state_dict(torch.load("identity.pth"))

    pl = pv.Plotter(window_size=(1080, 1080))
    pl.show(interactive_update=True)

    utils.setup_seed(seed)
    print("\tStep 1: Mapping to a semi-cube")
    T1 = time.time()
    optimize_semi_cube(0, pl, config, net, bound_remesh_scale, inner_scale)
    torch.save(net.state_dict(), os.path.join(save_dir, model_name + "_step1"))
    print("\tFinished, Time Cost = %.03f s" % (time.time() - T1))

    utils.setup_seed(seed)
    net.load_state_dict(torch.load(os.path.join(save_dir, model_name + "_step1")))
    print("\tStep 2: Mapping to unit square")
    T2 = time.time()
    optimize_cube(1, pl, config, net, bound_remesh_scale, inner_scale)
    torch.save(net.state_dict(), os.path.join(save_dir, model_name + "_step2"))
    print("\tFinished, Time Cost = %.03f s" % (time.time() - T2))

    utils.setup_seed(seed)
    net.load_state_dict(torch.load(os.path.join(save_dir, model_name + "_step2")))
    print("\tStep 3: Fix Corner Optimize")
    T3 = time.time()
    points_bound = np.copy(bound_remesh_scale.vertices).astype(np.float32)
    points_bound_torch = torch.from_numpy(points_bound).to(device)
    out_bound = net(points_bound_torch).detach().cpu().numpy()
    idx_corners = utils.sel_corners_dis(out_bound)
    optimize_cube_fixcorner(2, pl, config, net, bound_remesh_scale, inner_scale, idx_corners)
    torch.save(net.state_dict(), os.path.join(save_dir, model_name + "_step3"))
    print("\tFinished, Time Cost = %.03f s" % (time.time() - T3))

    utils.setup_seed(seed)
    net.load_state_dict(torch.load(os.path.join(save_dir, model_name + "_step3")))
    print("\tStep 4: Fix Edge Optimize")
    T4 = time.time()
    points_bound = np.copy(bound_remesh_scale.vertices).astype(np.float32)
    points_bound_torch = torch.from_numpy(points_bound).to(device)
    out_bound = net(points_bound_torch).detach().cpu().numpy()
    idx_corners = utils.sel_corners_dis(out_bound)
    V_mesh, facev_idx, facev_faceid, F, F_prop, edges_idx, edges_tar = utils.sel_edges(out_bound, bound_remesh_scale, idx_corners)
    # list 12, n*3
    edges_idx_cat = np.concatenate(edges_idx).astype(np.int32)
    Nv_edges = [len(edge) for edge in edges_idx]
    Index_edges_cat = np.cumsum(np.array([0, *Nv_edges], dtype=np.int32))
    edges_tar_cat = np.concatenate(edges_tar, axis=0).astype(np.float32)
    bound_edge = trimesh.Trimesh(V_mesh, F)
    optimize_cube_fixedge_cut(3, pl, config, net, bound_edge, inner, idx_corners,
                              [edges_idx_cat, edges_tar_cat, Index_edges_cat], [facev_idx, facev_faceid], F_prop)
    torch.save(net.state_dict(), os.path.join(save_dir, model_name + "_step4"))
    print("\tFinished, Time Cost = %.03f s" % (time.time() - T4))
    joblib.dump([V_mesh, F, F_prop, edges_idx, edges_tar], os.path.join(save_dir, model_name + "_edge.mesh"))

    utils.setup_seed(seed)
    net.load_state_dict(torch.load(os.path.join(save_dir, model_name + "_step4")))
    V_mesh_cut, F_cut, F_cut_prop, edges_idx, edges_tar = joblib.load(os.path.join(save_dir, model_name + "_edge.mesh"))
    print("\tStep 5: Spline Fitting and Optimize")
    T5 = time.time()
    with torch.no_grad():
        V_mesh_cut_map = net(torch.from_numpy(V_mesh_cut.astype(np.float32)).to(device)).detach().cpu().numpy()
    V_mesh_cut_map_adjust = utils.cube_map_adjust(V_mesh_cut_map, F_cut, F_cut_prop, edges_idx, edges_tar, method="optimize").astype(np.float32)

    print("\t\t5.1 Calc inverse")
    N_fitting = config.N_fitting
    V_bound = bound_remesh_scale.vertices
    V_inner, _, _, h = inner_scale
    sample_xyz = np.concatenate([V_mesh_cut, V_inner]).astype(np.float32)
    sample_xyz_torch = torch.from_numpy(sample_xyz).to(device)
    with torch.no_grad():
        sample_uvw = net(sample_xyz_torch).detach().cpu().numpy()
        # sample_uvw_torch[:V_mesh_edgecut.shape[0], :]= utils.get_tar_cube_torch(sample_uvw_torch[:V_mesh_edgecut.shape[0], :])
    rbf = RBFInterpolator(sample_uvw.reshape([-1, 3]), sample_xyz.reshape([-1, 3]), neighbors=30, kernel='linear')
    for p in net.parameters():
        p.requires_grad = False
    fitting_uvw = utils.get_uvw_bound(N_fitting).astype(np.float32).reshape([-1, 3])
    fitting_xyz_torch = torch.from_numpy(rbf(fitting_uvw).astype(np.float32)).to(device)
    fitting_uvw_torch = torch.from_numpy(fitting_uvw).to(device)
    fitting_uvw_torch_bound = torch.cat([torch.from_numpy(V_mesh_cut_map_adjust).to(device), fitting_uvw_torch])
    fitting_xyz_torch_bound = torch.cat([torch.from_numpy(V_mesh_cut.astype(np.float32)).to(device), fitting_xyz_torch])

    print("\t\t5.2 Spline fitting")
    bs_forward.fit(fitting_uvw_torch_bound, fitting_xyz_torch_bound, 2)
    print("\tFinished, Time Cost = %.03f s" % (time.time() - T5))
    torch.cuda.empty_cache()
    print("\t\t5.3 Spline Optimizing")
    T6 = time.time()
    for p in net.parameters():
        p.requires_grad = False
    optimize_bs(4, pl, config, bs_forward, V_mesh_cut_map_adjust, V_mesh_cut, F_cut)

    bs_forward.coeff.data = (bs_forward.coeff - 0.5) / scale + \
                            torch.from_numpy(center_point.astype(np.float32)).to(device)[:, None, None, None]
    print("\tFinished, Time Cost = %.03f s" % (time.time() - T6))
    print("\tAll Time Cost = %.03f s" % (time.time() - T_start))
    torch.save(bs_forward.state_dict(), os.path.join(save_dir, model_name + "_step5_bs"))

    bs_forward.load_state_dict(torch.load(os.path.join(save_dir, model_name + "_step5_bs")))
    print("\tVisualize")
    utils.pv_reset_plotter(pl)
    N_show = 257
    uvw = utils.get_uvw_bound(N_show).astype(np.float32)
    V_idx, F6 = utils.get_surface_triangles(N_show, N_show, N_show)
    uvw_bound = uvw[V_idx[:, 0], V_idx[:, 1], V_idx[:, 2], :]
    xyz_bound = bs_forward(torch.from_numpy(uvw_bound).to(device)).detach().cpu().numpy()
    face_colors = ["#ffbab0", "#82bae4", "#c6e7ab", "#ffc273", "#e4e4e4", "#99d4c9", "#d0bbd1", "#f8d1dd", "#ffbab0", "#a2ebf4"]
    pv_faces = [pv.make_tri_mesh(xyz_bound, F6[i]) for i in range(6)]
    for i in range(6):
        pl.add_mesh(pv_faces[i], color=face_colors[i], name=str(i))
    pl.reset_camera()
    pl.show()


if not os.path.exists("./img"):
    os.mkdir("./img")
if not os.path.exists("./result"):
    os.mkdir("./result")
config = Config()
config.name = "conch"
run(config)
