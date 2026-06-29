import os
import torch
import numpy as np
import math
from tqdm import trange
from scipy import constants
from concurrent.futures import ProcessPoolExecutor

from . import DATA_DIR

device = torch.device('cpu')
dtype = torch.float32

ABSC_DB = f"{DATA_DIR}/spectrum"
X_GRID = torch.tensor(np.linspace(0, 1, 5), dtype=dtype, device=device)
T_GRID = torch.tensor(np.arange(300.0, 3001.0, 100.0), dtype=dtype, device=device)

@torch.jit.script
def i_b_eta(temperature: torch.Tensor, eta: torch.Tensor, n: float = 1.0) -> torch.Tensor:
    """
    Black body radiation intensity as a function of temperature and wavenumber (eta).
    """
    C1 = 2 * math.pi * constants.h * constants.c**2 * 1e8
    C2 = constants.h * constants.c / constants.k * 1e2
    temp = temperature.unsqueeze(-1) if temperature.dim() == 0 else temperature
    e_b_lambda = C1 * eta**3 / (n**2 * (torch.exp(C2 * eta / (n * temp)) - 1.0))
    return e_b_lambda / math.pi

@torch.jit.script
def torch_interp(x, xp, fp):
    # x, xp, fp are 1D tensors on the same device
    # This is a simple linear interpolation similar to np.interp
    xp_min, xp_max = xp[0], xp[-1]
    x_clamped = torch.clamp(x, xp_min, xp_max)
    idx = torch.searchsorted(xp, x_clamped, right=True) - 1
    idx = idx.clamp(0, xp.size(0)-2)

    x0 = xp[idx]
    x1 = xp[idx+1]
    y0 = fp[idx]
    y1 = fp[idx+1]
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x_clamped - x0)

# @torch.jit.script
def to_same_grid(kappas):
    # kappas: list of 1D torch tensors
    max_n = max(k.shape[0] for k in kappas)
    if len(kappas) == 1:
        return kappas
    new_x = torch.linspace(0, 1, max_n, device=kappas[0].device, dtype=kappas[0].dtype)
    out = []
    for k in kappas:
        n = k.shape[0]
        if n < max_n:
            old_x = torch.linspace(0, 1, n, device=k.device, dtype=k.dtype)
            out.append(torch_interp(new_x, old_x, k))
        else:
            out.append(k)
    return out

def get_specie_kappas(specie: str, x: float, T: float):
    # 使用CPU加载数据，然后转到GPU
    x_grid = X_GRID.cpu().numpy()
    T_grid = T_GRID.cpu().numpy()

    def load_kappa(xv, Tv):
        k = np.load(os.path.join(ABSC_DB, "01.0", specie, f"{xv:.2f}_{int(Tv):04d}.npy"))
        return torch.from_numpy(k).to(device, dtype=dtype)

    id_x = np.searchsorted(x_grid, x, side='right') - 1
    id_T = np.searchsorted(T_grid, T, side='right') - 1

    if x == x_grid[id_x] and T == T_grid[id_T]:
        return load_kappa(x, T)
    elif x == x_grid[id_x]:
        Tl = T_grid[id_T]
        Tr = T_grid[id_T + 1]
        kl = load_kappa(x, Tl)
        kr = load_kappa(x, Tr)
        kl, kr = to_same_grid([kl, kr])
        return kl + (kr - kl) * (T - Tl) / (Tr - Tl)
    elif T == T_grid[id_T]:
        xl = x_grid[id_x]
        xr = x_grid[id_x + 1]
        kl = load_kappa(xl, T)
        kr = load_kappa(xr, T)
        kl, kr = to_same_grid([kl, kr])
        return kl + (kr - kl) * (x - xl) / (xr - xl)
    else:
        xl = x_grid[id_x]
        xr = x_grid[id_x + 1]
        Tl = T_grid[id_T]
        Tr = T_grid[id_T + 1]
        kll = load_kappa(xl, Tl)
        klr = load_kappa(xl, Tr)
        krl = load_kappa(xr, Tl)
        krr = load_kappa(xr, Tr)
        kll, klr, krl, krr = to_same_grid([kll, klr, krl, krr])
        kl = kll + (klr - kll) * (T - Tl) / (Tr - Tl)
        kr = krl + (krr - krl) * (T - Tl) / (Tr - Tl)
        return kl + (kr - kl) * (x - xl) / (xr - xl)

def get_mixture_kappas(xs: torch.Tensor, T: float):
    species = ["H2O", "CO2", "CO_"]
    x_cpu = xs.cpu().numpy()
    kappas_list = []
    for i, sp in enumerate(species):
        k = get_specie_kappas(sp, float(x_cpu[i]), float(T))
        kappas_list.append(k)
    max_n = max(k.shape[0] for k in kappas_list)
    mixture_kappas = torch.zeros(max_n, dtype=dtype, device=device)
    new_x = torch.linspace(0, 1, max_n, device=device, dtype=dtype)
    for i, kappa in enumerate(kappas_list):
        if kappa.shape[0] != max_n:
            old_x = torch.linspace(0, 1, kappa.shape[0], device=device, dtype=dtype)
            kappa = torch_interp(new_x, old_x, kappa)
        mixture_kappas += kappa * xs[i]
    return mixture_kappas

def get_kappas(xss: torch.Tensor, Ts: torch.Tensor):
    kappas_list = []
    for (xs, T) in zip(xss, Ts):
        k = get_mixture_kappas(xs, T.item())
        kappas_list.append(k)
    kappas_list = to_same_grid(kappas_list)
    return torch.stack(kappas_list, dim=0)

def get_i_b_etas(Ts: torch.Tensor, etas: torch.Tensor):
    return torch.stack([i_b_eta(T, etas) for T in Ts], dim=0)

# @torch.jit.script
# def e1(x):
#     mask = x <= 1.0
#     e1_val = torch.empty_like(x)

#     a0 = -0.57721566
#     a1 = 0.99999193
#     a2 = -0.24991055
#     a3 = 0.05519968
#     a4 = -0.00976004
#     a5 = 0.00107857

#     part1 = (a0 + x*(a1 + x*(a2 + x*(a3 + x*(a4 + x*a5)))) - torch.log(x+1e-8))
#     e1_val[mask] = part1[mask]

#     a1_ = 8.5733287401
#     a2_ = 18.0590169730
#     a3_ = 8.6347608925
#     a4_ = 0.2677737343
#     b1 = 9.5733223454
#     b2 = 25.6329561486
#     b3 = 21.0996530827
#     b4 = 3.9584969228
#     xm = x[~mask]
#     numerator = (xm*(a3_ + xm*(a2_ + xm*(a1_ + xm))) + a4_)
#     denominator = (xm*(b3 + xm*(b2 + xm*(b1 + xm))) + b4)
#     e1_val[~mask] = (numerator/denominator)*torch.exp(-xm)/xm

#     return e1_val

def e1(x):
    # Ensure x is positive and not too small
    x = torch.clamp(x, min=1e-6)  # Prevent negative or near-zero x

    # Coefficients for x <= 1.0
    a0 = -0.57721566
    a1 = 0.99999193
    a2 = -0.24991055
    a3 = 0.05519968
    a4 = -0.00976004
    a5 = 0.00107857
    part1 = (a0 + x * (a1 + x * (a2 + x * (a3 + x * (a4 + x * a5)))) - torch.log(x))

    # Coefficients for x > 1.0
    a1_ = 8.5733287401
    a2_ = 18.0590169730
    a3_ = 8.6347608925
    a4_ = 0.2677737343
    b1 = 9.5733223454
    b2 = 25.6329561486
    b3 = 21.0996530827
    b4 = 3.9584969228
    numerator = (x * (a3_ + x * (a2_ + x * (a1_ + x))) + a4_)
    denominator = (x * (b3 + x * (b2 + x * (b1 + x))) + b4)
    part2 = (numerator / denominator) * torch.exp(-x) / x

    mask = x <= 1.0
    e1_val = torch.where(mask, part1, part2)
    return e1_val

@torch.jit.script
def expn_float(n: int, x: torch.Tensor) -> torch.Tensor:
    if n < 1:
        raise ValueError("Order N must be at least 1.")
    en_val = e1(x)
    if n > 1:
        ex = torch.exp(-x)
        for i in range(2, n+1):
            en_val = (ex - x*en_val)/(i-1)
    return en_val

@torch.jit.script
def expn(n: int, xs: torch.Tensor) -> torch.Tensor:
    return expn_float(n, xs)

@torch.jit.script
def solve_rte(xs, kappas, I_bs, I_b1, I_b2):
    d_xs = xs[1:] - xs[:-1]
    d_taus = (kappas[1:] + kappas[:-1])/2.0 * d_xs
    taus = torch.zeros_like(xs)
    taus[1:] = torch.cumsum(d_taus, dim=0)
    tau_L = taus[-1]

    expn2 = torch.zeros((len(xs), len(xs)), device=xs.device, dtype=xs.dtype)
    expn3 = torch.zeros((len(xs), len(xs)), device=xs.device, dtype=xs.dtype)
    for i in range(len(xs)):
        expn2[i, :] = expn(2, torch.abs(taus - taus[i]))
        expn3[i, :] = expn(3, torch.abs(taus - taus[i]))

    G = torch.zeros_like(xs)
    term1 = I_b1 * expn(2, taus)
    term2 = I_b2 * expn(2, tau_L - taus)
    for i in range(len(xs)):
        integral_term1 = torch.trapz(I_bs[:i+1], expn2[i, :i+1])
        integral_term2 = torch.trapz(torch.flip(I_bs[i:], dims=[0]), torch.flip(expn2[i, i:], dims=[0]))
        G[i] = 2 * torch.pi * (term1[i] + term2[i] + integral_term1 + integral_term2)

    q = torch.zeros_like(xs)
    term1 = I_b1 * expn(3, taus)
    term2 = I_b2 * expn(3, tau_L - taus)
    for i in range(len(xs)):
        integral_term1 = torch.trapz(I_bs[:i+1], expn3[i, :i+1])
        integral_term2 = torch.trapz(torch.flip(I_bs[i:], dims=[0]), torch.flip(expn3[i, i:], dims=[0]))
        q[i] = 2 * torch.pi * (term1[i] - term2[i] + integral_term1 - integral_term2)
    dq = (4*math.pi*I_bs - G)*kappas
    return G, q, dq

@torch.jit.script
def solve_rte_batch(xs, kappas, I_bs, I_b1, I_b2):
    """
    Compute radiative transfer quantities for a batch of wavenumbers.

    Args:
        xs: Positions, shape (N,)
        kappas: Absorption coefficients, shape (N, B), where B is batch size
        I_bs: Blackbody intensities, shape (N, B)
        I_b1: Boundary condition, scalar
        I_b2: Boundary condition, scalar

    Returns:
        G: Incident radiation, shape (N, B)
        q: Flux, shape (N, B)
        dq: Flux derivative, shape (N, B)
    """
    # xs = xs.contiguous()
    # kappas = kappas.contiguous()
    # I_bs = I_bs.contiguous()
    # I_b1 = I_b1.contiguous()
    # I_b2 = I_b2.contiguous()
    N, B = kappas.shape
    device = xs.device
    dtype = xs.dtype

    # Compute position differences (independent of batch)
    d_xs = xs[1:] - xs[:-1]  # shape (N-1,)

    # Compute optical depth increments for all batch elements
    d_taus = (kappas[1:, :] + kappas[:-1, :]) / 2.0 * d_xs[:, None]  # shape (N-1, B)
    taus = torch.zeros((N, B), device=device, dtype=dtype)
    taus[1:, :] = torch.cumsum(d_taus, dim=0)  # shape (N, B)
    tau_L = taus[-1, :]  # shape (B,)

    # Compute exponential integral differences for all i, j, and batch elements
    # |taus[j, b] - taus[i, b]| for all i, j, b
    abs_diff_tau = torch.abs(taus[:, None, :] - taus[None, :, :])  # shape (N, N, B)
    expn2 = expn(2, abs_diff_tau)  # shape (N, N, B)
    expn3 = expn(3, abs_diff_tau)  # shape (N, N, B)

    # Initialize outputs
    G = torch.zeros((N, B), device=device, dtype=dtype)
    q = torch.zeros((N, B), device=device, dtype=dtype)

    # Compute boundary terms for G
    term1 = I_b1 * expn(2, taus)  # shape (N, B)
    # abs here is very important for avoiding -0.0000
    term2 = I_b2 * expn(2, torch.abs(tau_L - taus))  # shape (N, B)

    # Compute boundary terms for q
    term1_q = I_b1 * expn(3, taus)  # shape (N, B)
    # abs here is very important for avoiding -0.0000
    term2_q = I_b2 * expn(3, torch.abs(tau_L - taus))  # shape (N, B)

    # Compute integrations for all i and batch elements
    for i in range(N):
        # For G: Integrate from 0 to i and from i to N-1
        # First integral: ∫_0^xi I_b(x) * E_2(|tau(x) - tau(xi)|) dx
        # Mimic original code, which uses torch.trapz(I_bs[:i+1], expn2[i, :i+1])
        # Vectorize over batch dimension
        integral_term1 = torch.trapz(I_bs[:i+1, :], expn2[i, :i+1, :], dim=0)  # shape (B,)

        # Second integral: ∫_xi^xN I_b(x) * E_2(|tau(x) - tau(xi)|) dx
        # Original code uses flipped arrays, which may be incorrect (see notes below)
        # For consistency, we'll replicate it here
        integral_term2 = torch.trapz(
            torch.flip(I_bs[i:, :], dims=[0]),
            torch.flip(expn2[i, i:, :], dims=[0]),
            dim=0
        )  # shape (B,)

        G[i, :] = 2 * torch.pi * (term1[i, :] + term2[i, :] + integral_term1 + integral_term2)

        # For q: Similar integrations using expn3
        # First integral
        integral_term1 = torch.trapz(I_bs[:i+1, :], expn3[i, :i+1, :], dim=0)  # shape (B,)
        # Second integral
        integral_term2 = torch.trapz(
            torch.flip(I_bs[i:, :], dims=[0]),
            torch.flip(expn3[i, i:, :], dims=[0]),
            dim=0
        )  # shape (B,)

        q[i, :] = 2 * torch.pi * (term1_q[i, :] - term2_q[i, :] + integral_term1 - integral_term2)

    # Compute dq
    dq = (4 * torch.pi * I_bs - G) * kappas  # shape (N, B)

    return G, q, dq
