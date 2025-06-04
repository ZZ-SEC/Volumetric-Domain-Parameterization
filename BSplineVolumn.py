import torch
import numpy as np
from utils import get_uvw_bound
import numba
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr


@numba.jit(nopython=True)
def fill_matrix(N_U, idx_U, N_V, idx_V, N_W, idx_W, Nu, Nv, Nw, smooth=0.0):
    row = []
    col = []
    data = []
    N_points = N_U.shape[0]
    order = N_U.shape[1]
    for p in range(N_points):
        N_u, N_v, N_w, idx_u, idx_v, idx_w = N_U[p, :], N_V[p, :], N_W[p, :], idx_U[p], idx_V[p], idx_W[p]
        for ii in range(0, order):
            for jj in range(0, order):
                for kk in range(0, order):
                    i, j, k = ii + idx_u, jj + idx_v, kk + idx_w
                    idx = i * Nv * Nw + j * Nw + k
                    row.append(p)
                    col.append(idx)
                    data.append(N_u[ii] * N_v[jj] * N_w[kk])
    # Smooth
    N_constraint = 0
    for i in range(1, Nu - 1):
        for j in range(0, Nv):
            for k in range(0, Nw):
                row.append(N_points + N_constraint)
                col.append(i * Nv * Nw + j * Nw + k)
                data.append(1 * smooth)
                row.append(N_points + N_constraint)
                col.append((i - 1) * Nv * Nw + j * Nw + k)
                data.append(-0.5 * smooth)
                row.append(N_points + N_constraint)
                col.append((i + 1) * Nv * Nw + j * Nw + k)
                data.append(-0.5 * smooth)
                N_constraint += 1
    for i in range(0, Nu):
        for j in range(1, Nv - 1):
            for k in range(0, Nw):
                row.append(N_points + N_constraint)
                col.append(i * Nv * Nw + j * Nw + k)
                data.append(1 * smooth)
                row.append(N_points + N_constraint)
                col.append(i * Nv * Nw + (j - 1) * Nw + k)
                data.append(-0.5 * smooth)
                row.append(N_points + N_constraint)
                col.append(i * Nv * Nw + (j + 1) * Nw + k)
                data.append(-0.5 * smooth)
                N_constraint += 1
    for i in range(0, Nu):
        for j in range(0, Nv):
            for k in range(1, Nw - 1):
                row.append(N_points + N_constraint)
                col.append(i * Nv * Nw + j * Nw + k)
                data.append(1 * smooth)
                row.append(N_points + N_constraint)
                col.append(i * Nv * Nw + j * Nw + k - 1)
                data.append(-0.5 * smooth)
                row.append(N_points + N_constraint)
                col.append(i * Nv * Nw + j * Nw + k + 1)
                data.append(-0.5 * smooth)
                N_constraint += 1
    SQ2 = np.sqrt(2)
    for i in range(0, Nu - 1):
        for j in range(0, Nv - 1):
            for k in range(0, Nw):
                row.append(N_points + N_constraint)
                col.append(i * Nv * Nw + j * Nw + k)
                data.append(SQ2 / 2 * smooth)
                row.append(N_points + N_constraint)
                col.append((i + 1) * Nv * Nw + (j + 1) * Nw + k)
                data.append(SQ2 / 2 * smooth)
                row.append(N_points + N_constraint)
                col.append((i + 1) * Nv * Nw + j * Nw + k)
                data.append(-SQ2 / 2 * smooth)
                row.append(N_points + N_constraint)
                col.append(i * Nv * Nw + (j + 1) * Nw + k)
                data.append(-SQ2 / 2 * smooth)
                N_constraint += 1
    for i in range(0, Nu - 1):
        for j in range(0, Nv):
            for k in range(0, Nw - 1):
                row.append(N_points + N_constraint)
                col.append(i * Nv * Nw + j * Nw + k)
                data.append(SQ2 / 2 * smooth)
                row.append(N_points + N_constraint)
                col.append((i + 1) * Nv * Nw + j * Nw + k + 1)
                data.append(SQ2 / 2 * smooth)
                row.append(N_points + N_constraint)
                col.append((i + 1) * Nv * Nw + j * Nw + k)
                data.append(-SQ2 / 2 * smooth)
                row.append(N_points + N_constraint)
                col.append(i * Nv * Nw + j * Nw + k + 1)
                data.append(-SQ2 / 2 * smooth)
                N_constraint += 1
    for i in range(0, Nu):
        for j in range(0, Nv - 1):
            for k in range(0, Nw - 1):
                row.append(N_points + N_constraint)
                col.append(i * Nv * Nw + j * Nw + k)
                data.append(SQ2 / 2 * smooth)
                row.append(N_points + N_constraint)
                col.append(i * Nv * Nw + (j + 1) * Nw + k + 1)
                data.append(SQ2 / 2 * smooth)
                row.append(N_points + N_constraint)
                col.append(i * Nv * Nw + (j + 1) * Nw + k)
                data.append(-SQ2 / 2 * smooth)
                row.append(N_points + N_constraint)
                col.append(i * Nv * Nw + j * Nw + k + 1)
                data.append(-SQ2 / 2 * smooth)
                N_constraint += 1
    return data, row, col, N_points + N_constraint


def N_Torch(X, knots, p, require_D=False):
    idx = torch.searchsorted(knots, X, side='left') - 1
    idx -= p
    Np = X.shape[0]
    Mat = torch.zeros((Np, p + 1, p + 1), dtype=X.dtype, device=X.device)
    Mat[:, 0, p] = 1
    knot = torch.empty((Np, 2 * p + 1), dtype=X.dtype, device=X.device)
    for i in range(2 * p + 1):
        knot[:, i] = knots[idx + i]
    for k in range(p):
        for i in range(p):
            Mat[:, k + 1, i] = (X - knot[:, i]) / (knot[:, i + k + 1] - knot[:, i] + 1e-20) * Mat[:, k, i] \
                               + (knot[:, i + k + 2] - X) / (knot[:, i + k + 2] - knot[:, i + 1] + 1e-20) * Mat[:, k, i + 1]
        Mat[:, k + 1, p] = (X - knot[:, p]) / (knot[:, p + k + 1] - knot[:, p] + 1e-20) * Mat[:, k, p]
    if require_D:
        DN = torch.zeros((Np, p + 1), dtype=X.dtype, device=X.device)
        for i in range(p):
            DN[:, i] = p / (knot[:, p + i] - knot[:, i] + 1e-20) * Mat[:, -2, i] \
                       - p / (knot[:, p + i + 1] - knot[:, i + 1] + 1e-20) * Mat[:, -2, i + 1]
        DN[:, p] = p / (knot[:, p + p] - knot[:, p] + 1e-20) * Mat[:, -2, p]
    else:
        DN = None
    return (Mat[:, -1, :]).contiguous(), DN, idx


def Calc_Torch(uvw, knotx, knoty, knotz, coeff, order, requires_D=False):
    xyz = torch.zeros_like(uvw)
    U, V, W = uvw[0, :], uvw[1, :], uvw[2, :]
    N_U, DN_U, idx_U = N_Torch(U, knotx, order - 1, requires_D)
    N_V, DN_V, idx_V = N_Torch(V, knoty, order - 1, requires_D)
    N_W, DN_W, idx_W = N_Torch(W, knotz, order - 1, requires_D)
    for ii in range(0, order):
        for jj in range(0, order):
            for kk in range(0, order):
                i, j, k = ii + idx_U, jj + idx_V, kk + idx_W
                xyz += coeff[:, i, j, k] * (N_U[:, ii] * N_V[:, jj] * N_W[:, kk]).reshape([1, -1])
    return xyz, N_U, DN_U, idx_U, N_V, DN_V, idx_V, N_W, DN_W, idx_W


def Calc_D_Torch(grad_out, N_U, DN_U, idx_U, N_V, DN_V, idx_V, N_W, DN_W, idx_W, coeff, order):
    grad_uvw = torch.zeros_like(grad_out)
    grad_coeff = torch.zeros_like(coeff)
    _, Nu, Nv, Nw = coeff.shape
    for ii in range(0, order):
        for jj in range(0, order):
            for kk in range(0, order):
                i, j, k = ii + idx_U, jj + idx_V, kk + idx_W
                temp = (grad_out * coeff[:, i, j, k]).sum(0)
                grad_uvw[0, :] += temp * DN_U[:, ii] * N_V[:, jj] * N_W[:, kk]
                grad_uvw[1, :] += temp * N_U[:, ii] * DN_V[:, jj] * N_W[:, kk]
                grad_uvw[2, :] += temp * N_U[:, ii] * N_V[:, jj] * DN_W[:, kk]
                temp = grad_out * (N_U[:, ii] * N_V[:, jj] * N_W[:, kk]).reshape([1, -1])
                index_flatten = i * Nv * Nw + j * Nw + k
                grad_coeff = grad_coeff.reshape([-1])
                grad_coeff.data.scatter_add_(0, index_flatten, temp[0, :])
                grad_coeff.data.scatter_add_(0, index_flatten + Nu * Nv * Nw, temp[1, :])
                grad_coeff.data.scatter_add_(0, index_flatten + Nu * Nv * Nw * 2, temp[2, :])
                grad_coeff = grad_coeff.reshape(coeff.shape)
    return grad_uvw, grad_coeff


class BS_Func_Torch_(torch.autograd.Function):
    @staticmethod
    def forward(self, uvw, knotx, knoty, knotz, coeff, order):
        with torch.no_grad():
            xyz, N_U, DN_U, idx_U, N_V, DN_V, idx_V, N_W, DN_W, idx_W = Calc_Torch(uvw, knotx, knoty, knotz, coeff, order, requires_D=True)
        self.save_for_backward(N_U, DN_U, idx_U, N_V, DN_V, idx_V, N_W, DN_W, idx_W, coeff, order)

        return xyz

    @staticmethod
    def backward(self, grad_output):
        N_U, DN_U, idx_U, N_V, DN_V, idx_V, N_W, DN_W, idx_W, coeff, order = self.saved_tensors
        with torch.no_grad():
            grad_uvw, grad_coeff = Calc_D_Torch(grad_output, N_U, DN_U, idx_U, N_V, DN_V, idx_V, N_W, DN_W, idx_W, coeff, order)
        return grad_uvw, None, None, None, grad_coeff, None


def BS_Func_Torch(uvw, knotx, knoty, knotz, coeff, order):
    uvw = torch.clip(uvw, min=0 + 1e-14, max=1 - 1e-14)
    xyz = BS_Func_Torch_.apply(uvw, knotx, knoty, knotz, coeff, torch.tensor(order, device=uvw.device))
    return xyz


class BS_Torch(torch.nn.Module):
    def __init__(self, Nu=30, Nv=30, Nw=30, order=3, coeff=None):
        super().__init__()
        self.order = order

        p = order - 1
        if coeff is not None:
            Nu, Nv, Nw = coeff.shape[1], coeff.shape[2], coeff.shape[3]
        self.Nu = Nu
        self.Nv = Nv
        self.Nw = Nw
        knotx = torch.cat([torch.zeros([p]), torch.linspace(0, 1, Nu - p + 1), torch.ones([p])])
        knoty = torch.cat([torch.zeros([p]), torch.linspace(0, 1, Nv - p + 1), torch.ones([p])])
        knotz = torch.cat([torch.zeros([p]), torch.linspace(0, 1, Nw - p + 1), torch.ones([p])])
        self.p = p
        self.knotx = torch.nn.Parameter(knotx, requires_grad=False)
        self.knoty = torch.nn.Parameter(knoty, requires_grad=False)
        self.knotz = torch.nn.Parameter(knotz, requires_grad=False)
        if coeff is None:
            coeff = torch.from_numpy(get_uvw_bound(Nu, Nv, Nw).astype(np.float32)).permute([3, 0, 1, 2]).contiguous()
        self.coeff = torch.nn.Parameter(coeff, requires_grad=True)

    def load_coeff(self, coeff):
        Nu, Nv, Nw = self.Nu, self.Nv, self.Nw
        device = coeff.device
        dtype = coeff.dtype
        p = self.p
        knotx = torch.cat([torch.zeros([p]), torch.linspace(0, 1, Nu - p + 1), torch.ones([p])]).type(dtype).to(device)
        knoty = torch.cat([torch.zeros([p]), torch.linspace(0, 1, Nv - p + 1), torch.ones([p])]).type(dtype).to(device)
        knotz = torch.cat([torch.zeros([p]), torch.linspace(0, 1, Nw - p + 1), torch.ones([p])]).type(dtype).to(device)
        self.knotx.data = knotx
        self.knoty.data = knoty
        self.knotz.data = knotz
        self.coeff.data = coeff

    def __call__(self, uvw_):
        shape = uvw_.shape
        uvw_ = uvw_.reshape([-1, 3])
        uvw = uvw_.permute([1, 0]).contiguous()
        xyz = BS_Func_Torch(uvw, self.knotx, self.knoty, self.knotz, self.coeff, self.order)
        xyz = xyz.permute([1, 0])
        return xyz.reshape(shape)

    def fit(self, uvw, xyz, smooth=0.01):
        uvw = uvw.reshape([-1, 3])
        xyz = xyz.reshape([-1, 3])
        uvw = torch.clip(uvw, min=0 + 1e-16, max=1 - 1e-16)
        U, V, W = uvw[:, 0], uvw[:, 1], uvw[:, 2]
        with torch.no_grad():
            N_U, _, idx_U = N_Torch(U.contiguous(), self.knotx, self.order - 1)
            N_V, _, idx_V = N_Torch(V.contiguous(), self.knoty, self.order - 1)
            N_W, _, idx_W = N_Torch(W.contiguous(), self.knotz, self.order - 1)
            N_U, N_V, N_W, idx_U, idx_V, idx_W = N_U.detach().cpu().numpy().astype(np.float64), N_V.detach().cpu().numpy().astype(
                np.float64), N_W.detach().cpu().numpy().astype(np.float64), idx_U.detach().cpu().numpy().astype(
                np.int32), idx_V.detach().cpu().numpy().astype(np.int32), idx_W.detach().cpu().numpy().astype(np.int32)
        data, row, col, N_constraint = fill_matrix(N_U, idx_U, N_V, idx_V, N_W, idx_W, self.Nu, self.Nv, self.Nw, smooth)
        N_points = uvw.shape[0]
        N_coeff = self.Nu * self.Nv * self.Nw
        M = csr_matrix((data, (row, col)), shape=(N_constraint, N_coeff), dtype=np.float32)
        xyz = xyz.detach().cpu().numpy()
        if N_constraint > N_points:
            b = np.concatenate([xyz, np.zeros([N_constraint - N_points, 3])], axis=0).astype(np.float32)
        else:
            b = xyz

        coeff_1 = lsmr(M, b[:, 0])[0]
        coeff_2 = lsmr(M, b[:, 1])[0]
        coeff_3 = lsmr(M, b[:, 2])[0]
        coeff = np.concatenate([coeff_1.reshape([1, -1]), coeff_2.reshape([1, -1]), coeff_3.reshape([1, -1])], axis=0). \
            reshape([3, self.Nu, self.Nv, self.Nw])
        device = self.coeff.data.device
        self.coeff.data = torch.from_numpy(coeff.astype(np.float32)).to(device)
