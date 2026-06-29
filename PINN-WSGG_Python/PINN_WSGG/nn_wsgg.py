import torch
from scipy.constants import Stefan_Boltzmann
from .rte import solve_rte
from .mlp import MLP


class NN_WSGG(torch.nn.Module):
    """
    NN-WSGG model with pressure input.

    Input:
        [T, x_H2O, x_CO2, x_CO, p]

    Output:
        first 9 outputs  -> weights a0, a1, ..., a8
        last 8 outputs   -> absorption coefficients k1, ..., k8

    Notes:
        a0 is the transparent gas weight.
        k0 = 0.
        A single shared MLP is used, so T and p both affect weights and kappa.

    Important:
        For NN input normalization:
            p is transformed as log(p).

        For RTE physical calculation:
            raw p is still used.
    """

    def __init__(self, hidden_dims, x_mean: torch.Tensor, x_std: torch.Tensor):
        super().__init__()

        # Single shared MLP:
        # input:  [T, x_H2O, x_CO2, x_CO, log(p)]
        # output: 17 = 9 weights + 8 absorption coefficients
        self.mlp = MLP([5, *hidden_dims, 17])

        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)

    def forward(self, x):
        """
        x shape: [N, 5]

        x[:, 0] = T
        x[:, 1] = x_H2O
        x[:, 2] = x_CO2
        x[:, 3] = x_CO
        x[:, 4] = p

        Note:
            The input pressure p is converted to log(p)
            before standardization.
        """

        x = x.clone()

        # Use log(p) only for neural-network input normalization
        x[:, 4] = torch.log(x[:, 4].clamp_min(1e-8))

        x_norm = (x - self.x_mean) / self.x_std

        out = self.mlp(x_norm)

        # First 9 outputs: weights a0~a8
        weights_raw = out[:, :9]

        # Last 8 outputs: non-transparent absorption coefficients k1~k8
        abscs_raw = out[:, 9:]

        # weights satisfy a0 + a1 + ... + a8 = 1
        weights = torch.softmax(weights_raw, dim=1)

        # k0 = 0, k1~k8 > 0
        abscs = torch.zeros_like(weights)
        abscs[:, 1:] = torch.exp(abscs_raw)

        return weights, abscs


# ============================================================
# Emissivity calculation
# No pressure multiplication in epsilon
# ============================================================
def nn_wsgg_epsilon(
    nn_wsgg,
    temperature,
    x_h2o,
    x_co2,
    x_co,
    pressure,
    L
):
    """
    Calculate NN-WSGG total emissivity.

    Important:
        The neural network input includes pressure.
        In NN_WSGG.forward(), pressure is converted to log(p)
        before standardization.

        However, the emissivity exponential term does not multiply pressure.
        It only uses x_sum * L.

    Inputs:
        nn_wsgg: NN_WSGG model
        temperature: gas temperature, shape [N] or [N, 1]
        x_h2o: H2O mole fraction, shape [N] or [N, 1]
        x_co2: CO2 mole fraction, shape [N] or [N, 1]
        x_co: CO mole fraction, shape [N] or [N, 1]
        pressure: raw pressure input for NN, shape [N] or [N, 1]
        L: path length, shape [N] or [N, 1]
    """

    if temperature.dim() == 1:
        temperature = temperature.reshape(-1, 1)
    if x_h2o.dim() == 1:
        x_h2o = x_h2o.reshape(-1, 1)
    if x_co2.dim() == 1:
        x_co2 = x_co2.reshape(-1, 1)
    if x_co.dim() == 1:
        x_co = x_co.reshape(-1, 1)
    if pressure.dim() == 1:
        pressure = pressure.reshape(-1, 1)
    if L.dim() == 1:
        L = L.reshape(-1, 1)

    # NN input includes raw pressure.
    # log(p) is handled inside NN_WSGG.forward().
    nn_input = torch.cat(
        [temperature, x_h2o, x_co2, x_co, pressure],
        dim=1
    )

    weights, abscs = nn_wsgg(nn_input)

    x_sum = x_h2o + x_co2 + x_co

    # Emissivity does NOT multiply pressure here
    epsilon = torch.sum(
        weights * (1.0 - torch.exp(-abscs * x_sum * L)),
        dim=1
    )

    return epsilon


# ============================================================
# RTE calculation
# Pressure is multiplied in RTE
# ============================================================
def nn_wsgg_rte(
    nn_wsgg,
    xs,
    temperature_profiles,
    x_h2o_profiles,
    x_co2_profiles,
    x_co_profiles,
    pressure_profiles,
    T_l,
    T_r
):
    """
    Calculate NN-WSGG RTE results.

    Inputs:
        xs: spatial coordinate
        temperature_profiles: T profile
        x_h2o_profiles: H2O mole fraction profile
        x_co2_profiles: CO2 mole fraction profile
        x_co_profiles: CO mole fraction profile
        pressure_profiles: raw pressure profile
        T_l: left boundary temperature
        T_r: right boundary temperature

    Important:
        NN input uses log(p) internally.
        RTE effective absorption coefficient uses raw p.
    """

    # NN input uses raw pressure here.
    # log(p) is handled inside NN_WSGG.forward().
    nn_input = torch.vstack([
        temperature_profiles,
        x_h2o_profiles,
        x_co2_profiles,
        x_co_profiles,
        pressure_profiles
    ]).T

    wsgg_a, wsgg_k = nn_wsgg(nn_input)

    gas_sum = (
        x_h2o_profiles
        + x_co2_profiles
        + x_co_profiles
    ).reshape(-1, 1)

    # Use raw pressure for physical absorption scaling
    pressure = pressure_profiles.reshape(-1, 1)

    # RTE uses pressure multiplication
    wsgg_k = wsgg_k * gas_sum * pressure

    wsgg_G = 0.0
    wsgg_q = 0.0
    wsgg_dq = 0.0

    for i in range(9):
        a = wsgg_a[:, i]

        # Unit conversion is kept
        k = wsgg_k[:, i] * 100.0

        i_b = (
            Stefan_Boltzmann
            * (temperature_profiles ** 4)
            * a
            / torch.pi
        )

        i_b1 = (
            Stefan_Boltzmann
            * (T_l ** 4)
            * a[0]
            / torch.pi
        )

        i_b2 = (
            Stefan_Boltzmann
            * (T_r ** 4)
            * a[-1]
            / torch.pi
        )

        cur_G, cur_q, cur_dq = solve_rte(
            xs,
            k,
            i_b,
            i_b1,
            i_b2
        )

        wsgg_G = wsgg_G + cur_G
        wsgg_q = wsgg_q + cur_q
        wsgg_dq = wsgg_dq + cur_dq

    return wsgg_G, wsgg_q, wsgg_dq


# ============================================================
# Epsilon loss
# New interface: pass nn_wsgg directly
# ============================================================
class EPSILON_LOSS(torch.nn.Module):
    """
    Pure MSE loss for emissivity.

    The NN input includes:
        [T, x_H2O, x_CO2, x_CO, p]

    Inside NN_WSGG.forward():
        p is converted to log(p).

    The emissivity exponential term uses:
        x_sum * L

    It does not multiply pressure in the exponential term.
    """

    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, nn_wsgg, loss_additional, lbl_epsilons):
        """
        loss_additional shape: [B, 6]

        column 0: T
        column 1: x_H2O
        column 2: x_CO2
        column 3: x_CO
        column 4: p
        column 5: L
        """

        temperature = loss_additional[:, 0:1]
        x_h2o = loss_additional[:, 1:2]
        x_co2 = loss_additional[:, 2:3]
        x_co = loss_additional[:, 3:4]
        pressure = loss_additional[:, 4:5]
        L = loss_additional[:, 5:6]

        eps_pred = nn_wsgg_epsilon(
            nn_wsgg,
            temperature,
            x_h2o,
            x_co2,
            x_co,
            pressure,
            L
        ).reshape(-1, 1)

        return self.mse(eps_pred, lbl_epsilons)
# import torch
# from scipy.constants import Stefan_Boltzmann
# from .rte import solve_rte
# from .mlp import MLP


# class NN_WSGG(torch.nn.Module):
#     """
#     NN-WSGG 模型（传统乘 p 版本）
#     输入: [T, x_h2o, x_co2, x_co, p]   ← p 为原始压力 (bar)
#     输出: weights 和 abscs（吸收系数不包含压力，压力在 epsilon 和 rte 中单独乘）
#     """
#     def __init__(self, hidden_dims, x_mean: torch.Tensor, x_std: torch.Tensor):
#         super().__init__()

#         # weights 使用 5 个输入（T + 3组分 + p）
#         self.mlp_weights = MLP([5, *hidden_dims, 9])
        
#         # abscs 使用 4 个输入（3组分 + p），不使用 T
#         self.mlp_abscs = MLP([4, *hidden_dims, 8])

#         self.register_buffer("x_mean", x_mean)
#         self.register_buffer("x_std", x_std)

#     def forward(self, x):
#         """
#         x: shape [N, 5] = [T, x_h2o, x_co2, x_co, p]   ← p 是原始压力
#         """
#         x = x.clone()
        
#         # ==================== log(p) 处理（只做一次） ====================
#         # dataset 中已经基于 log(p) 计算了 mean/std
#         x[:, 4] = torch.log(x[:, 4].clamp_min(1e-8))

#         # 标准化
#         x = (x - self.x_mean) / self.x_std

#         # ====================== 前向计算 ======================
#         weights = torch.softmax(self.mlp_weights(x), dim=1)

#         # abscs: 第0个为透明气体（k=0），其余8个由网络预测
#         abscs = torch.zeros_like(weights)
#         abscs[:, 1:] = torch.exp(self.mlp_abscs(x[:, 1:]))

#         return weights, abscs


# # ====================== 发射率计算函数（传统乘 p） ======================
# def nn_wsgg_epsilon(weights, abscs, xs, p, L):
#     """
#     计算总发射率 epsilon（传统公式，乘压力 p）
    
#     参数:
#         weights: [B, 9]
#         abscs:   [B, 9]     ← 网络预测的吸收系数（不含压力）
#         xs:      [B, 3]     ← [x_h2o, x_co2, x_co]
#         p:       [B, 1]     ← 压力 (bar)
#         L:       [B, 1]     ← 光程长度 (m)
#     """
#     sum_xs = torch.sum(xs, dim=1, keepdim=True)
#     # 传统公式：光学厚度 = k * sum_x * p * L
#     epsilon = torch.sum(
#         weights * (1.0 - torch.exp(-abscs * sum_xs * p * L)), 
#         dim=1
#     )
#     return epsilon


# # ====================== RTE 计算函数（保持一致） ======================
# def nn_wsgg_rte(
#     nn_wsgg,
#     xs,
#     temperature_profiles,
#     x_h2o_profiles,
#     x_co2_profiles,
#     x_co_profiles,
#     pressure_profiles,
#     T_l,
#     T_r
# ):
#     """
#     基于 NN-WSGG 求解 RTE
#     """
#     nn_input = torch.vstack([
#         temperature_profiles,
#         x_h2o_profiles,
#         x_co2_profiles,
#         x_co_profiles,
#         pressure_profiles
#     ]).T

#     wsgg_a, wsgg_k = nn_wsgg(nn_input)

#     # 只乘参与辐射气体的组分和（不乘 pressure，因为 abscs 不包含压力）
#     gas_sum = (x_h2o_profiles + x_co2_profiles + x_co_profiles).reshape(-1, 1)
#     wsgg_k = wsgg_k * gas_sum

#     wsgg_G = 0.0
#     wsgg_q = 0.0
#     wsgg_dq = 0.0

#     for i in range(9):
#         a = wsgg_a[:, i]
#         k = wsgg_k[:, i] * 100.0                     # 注意：这个 *100.0 是否合理需要后续验证

#         i_b = Stefan_Boltzmann * (temperature_profiles ** 4) * a / torch.pi
#         i_b1 = Stefan_Boltzmann * (T_l ** 4) * a[0] / torch.pi
#         i_b2 = Stefan_Boltzmann * (T_r ** 4) * a[-1] / torch.pi

#         cur_G, cur_q, cur_dq = solve_rte(xs, k, i_b, i_b1, i_b2)

#         wsgg_G += cur_G
#         wsgg_q += cur_q
#         wsgg_dq += cur_dq

#     return wsgg_G, wsgg_q, wsgg_dq


# # ====================== 损失函数 ======================
# class EPSILON_LOSS(torch.nn.Module):
#     """
#     纯 MSE 损失（配合传统乘 p 的 nn_wsgg_epsilon 使用）
#     """
#     def __init__(self):
#         super().__init__()
#         self.mse = torch.nn.MSELoss()

#     def forward(self, weights, abscs, loss_additional, lbl_epsilons):
#         """
#         loss_additional: [B, 5] = [x_h2o, x_co2, x_co, p, L]
#         """
#         xs = loss_additional[:, 0:3]
#         p  = loss_additional[:, 3:4]
#         L  = loss_additional[:, 4:5]

#         eps_pred = nn_wsgg_epsilon(weights, abscs, xs, p, L).reshape(-1, 1)

#         return self.mse(eps_pred, lbl_epsilons)
# import torch
# from scipy.constants import Stefan_Boltzmann
# from .rte import solve_rte
# from .mlp import MLP


# class NN_WSGG(torch.nn.Module):
#     """
#     NN-WSGG 模型（最终干净版）
#     输入: [T, x_h2o, x_co2, x_co, p]   ← p 为原始压力 (bar)
#     内部只做一次 log(p) + 标准化
#     """
#     def __init__(self, hidden_dims, x_mean: torch.Tensor, x_std: torch.Tensor):
#         super().__init__()

#         # weights 使用 5 个输入（T + 3组分 + log(p)）
#         self.mlp_weights = MLP([5, *hidden_dims, 9])
        
#         # abscs 使用 4 个输入（3组分 + log(p)），不使用 T
#         self.mlp_abscs = MLP([4, *hidden_dims, 8])

#         self.register_buffer("x_mean", x_mean)
#         self.register_buffer("x_std", x_std)

#     def forward(self, x):
#         """
#         x: shape [N, 5] = [T, x_h2o, x_co2, x_co, p]   ← p 是原始压力
#         """
#         x = x.clone()
        
#         # ==================== 关键修改：只在这里做一次 log(p) ====================
#         # dataset 中已经基于 log(p) 计算了 mean/std，这里不再重复 log
#         x[:, 4] = torch.log(x[:, 4].clamp_min(1e-8))   # log(p)

#         # 标准化（使用 dataset 提供的 log(p) 统计量）
#         x = (x - self.x_mean) / self.x_std

#         # ====================== 前向计算 ======================
#         weights = torch.softmax(self.mlp_weights(x), dim=1)

#         # abscs: 第0个为透明气体（k=0），其余8个由网络预测
#         abscs = torch.zeros_like(weights)
#         abscs[:, 1:] = torch.exp(self.mlp_abscs(x[:, 1:]))

#         return weights, abscs


# # ====================== epsilon 计算函数 ======================
# def nn_wsgg_epsilon(weights, abscs, xs, L):
#     """
#     计算总发射率 epsilon
    
#     参数:
#         weights: [B, 9]
#         abscs:   [B, 9]   ← 已包含压力效应的等效吸收系数
#         xs:      [B, 3]   ← [x_h2o, x_co2, x_co]
#         L:       [B, 1]   ← 光程长度（单位：m）
#     """
#     sum_xs = torch.sum(xs, dim=1, keepdim=True)
#     epsilon = torch.sum(
#         weights * (1.0 - torch.exp(-abscs * sum_xs * L)), 
#         dim=1
#     )
#     return epsilon

# def nn_wsgg_rte(
#     nn_wsgg,
#     xs,
#     temperature_profiles,
#     x_h2o_profiles,
#     x_co2_profiles,
#     x_co_profiles,
#     pressure_profiles,
#     T_l,
#     T_r
# ):
#     """
#     基于 NN-WSGG 求解 RTE

#     参数:
#         nn_wsgg:              网络模型
#         xs:                   空间坐标
#         temperature_profiles: [N]
#         x_h2o_profiles:       [N]
#         x_co2_profiles:       [N]
#         x_co_profiles:        [N]
#         pressure_profiles:    [N]，传入原始 pressure
#         T_l:                  左边界温度
#         T_r:                  右边界温度

#     说明:
#         - pressure 通过网络输入影响 weights 和 abscs
#         - abscs 已包含压强效应，因此这里不要再额外乘 pressure_profiles
#         - 这里只额外乘参与辐射的组分和
#     """
#     nn_input = torch.vstack([
#         temperature_profiles,
#         x_h2o_profiles,
#         x_co2_profiles,
#         x_co_profiles,
#         pressure_profiles
#     ]).T

#     wsgg_a, wsgg_k = nn_wsgg(nn_input)

#     gas_sum = (x_h2o_profiles + x_co2_profiles + x_co_profiles).reshape(-1, 1)
#     wsgg_k = wsgg_k * gas_sum

#     wsgg_G = 0.0
#     wsgg_q = 0.0
#     wsgg_dq = 0.0

#     for i in range(9):
#         a = wsgg_a[:, i]
#         k = wsgg_k[:, i] * 100.0

#         i_b = Stefan_Boltzmann * (temperature_profiles ** 4) * a / torch.pi
#         i_b1 = Stefan_Boltzmann * (T_l ** 4) * a[0] / torch.pi
#         i_b2 = Stefan_Boltzmann * (T_r ** 4) * a[-1] / torch.pi

#         cur_G, cur_q, cur_dq = solve_rte(xs, k, i_b, i_b1, i_b2)

#         wsgg_G += cur_G
#         wsgg_q += cur_q
#         wsgg_dq += cur_dq

#     return wsgg_G, wsgg_q, wsgg_dq
# # ====================== 损失函数（保持最简洁） ======================
# class EPSILON_LOSS(torch.nn.Module):
#     """
#     纯 MSE 损失（按你的要求，未加入相对误差项）
#     """
#     def __init__(self):
#         super().__init__()
#         self.mse = torch.nn.MSELoss()

#     def forward(self, weights, abscs, loss_additional, lbl_epsilons):
#         """
#         loss_additional: [B, 4] = [x_h2o, x_co2, x_co, L]
#         """
#         eps_pred = nn_wsgg_epsilon(
#             weights, 
#             abscs, 
#             loss_additional[:, 0:3], 
#             loss_additional[:, 3:4]
#         ).reshape(-1, 1)

#         return self.mse(eps_pred, lbl_epsilons)
# import torch
# from scipy.constants import Stefan_Boltzmann
# from .rte import solve_rte
# from .mlp import MLP


# class NN_WSGG(torch.nn.Module):
#     """
#     输入:
#         x = [T, x_h2o, x_co2, x_co, p]

#     输出:
#         weights: 9个权重（8个灰气体 + 1个透明气体）
#         abscs:   9个吸收系数，其中第0个透明气体固定为0，其余8个由网络预测

#     说明:
#         - 权重受温度、组分、压强共同影响
#         - 吸收系数 abscs 是等效吸收系数，已包含压强效应
#         - 因此后处理时不要再额外乘 pressure
#         - pressure 在 forward 内部先做 log(p)，再做标准化
#           所以 x_mean / x_std 的第 5 列必须基于 log(p) 统计
#     """
#     def __init__(self, hidden_dims, x_mean: torch.Tensor, x_std: torch.Tensor):
#         super(NN_WSGG, self).__init__()

#         # 输入: [T, x_h2o, x_co2, x_co, log(p)归一化后]
#         self.mlp_weights = MLP([5, *hidden_dims, 9])

#         # 输入: [x_h2o, x_co2, x_co, log(p)归一化后]
#         self.mlp_abscs = MLP([4, *hidden_dims, 8])

#         self.register_buffer("x_mean", x_mean)
#         self.register_buffer("x_std", x_std)

#     def forward(self, x):
#         """
#         参数:
#             x: shape [N, 5]
#                每列依次为 [T, x_h2o, x_co2, x_co, p]
#                注意这里传入的是原始 pressure，不是 log(p)

#         返回:
#             weights: shape [N, 9]
#             abscs:   shape [N, 9]
#         """
#         x = x.clone()

#         # 关键修改：pressure 先取 log，再做标准化
#         x[:, 4] = torch.log(x[:, 4].clamp_min(1e-8))

#         x = (x - self.x_mean) / self.x_std

#         weights = torch.softmax(self.mlp_weights(x), dim=1)

#         abscs = torch.zeros_like(weights)
#         abscs[:, 1:] = torch.exp(self.mlp_abscs(x[:, 1:]))

#         return weights, abscs


# def nn_wsgg_epsilon(weights, abscs, xs, L):
#     """
#     计算总发射率 epsilon

#     参数:
#         weights: [B, 9]
#         abscs:   [B, 9]，已包含压强效应的等效吸收系数
#         xs:      [B, 3] -> [x_h2o, x_co2, x_co]
#         L:       [B, 1] -> 光程长度

#     说明:
#         abscs 已经包含压强效应，
#         因此这里不能再使用 pL，只使用 L
#     """
#     sum_xs = torch.sum(xs, dim=1, keepdim=True)
#     epsilon = torch.sum(
#         weights * (1.0 - torch.exp(-abscs * sum_xs * L)),
#         dim=1
#     )
#     return epsilon


# def nn_wsgg_rte(
#     nn_wsgg,
#     xs,
#     temperature_profiles,
#     x_h2o_profiles,
#     x_co2_profiles,
#     x_co_profiles,
#     pressure_profiles,
#     T_l,
#     T_r
# ):
#     """
#     基于 NN-WSGG 求解 RTE

#     参数:
#         nn_wsgg:              网络模型
#         xs:                   空间坐标
#         temperature_profiles: [N]
#         x_h2o_profiles:       [N]
#         x_co2_profiles:       [N]
#         x_co_profiles:        [N]
#         pressure_profiles:    [N]，传入原始 pressure
#         T_l:                  左边界温度
#         T_r:                  右边界温度

#     说明:
#         - pressure 通过网络输入影响 weights 和 abscs
#         - abscs 已包含压强效应，因此这里不要再额外乘 pressure_profiles
#         - 这里只额外乘参与辐射的组分和
#     """
#     nn_input = torch.vstack([
#         temperature_profiles,
#         x_h2o_profiles,
#         x_co2_profiles,
#         x_co_profiles,
#         pressure_profiles
#     ]).T

#     wsgg_a, wsgg_k = nn_wsgg(nn_input)

#     gas_sum = (x_h2o_profiles + x_co2_profiles + x_co_profiles).reshape(-1, 1)
#     wsgg_k = wsgg_k * gas_sum

#     wsgg_G = 0.0
#     wsgg_q = 0.0
#     wsgg_dq = 0.0

#     for i in range(9):
#         a = wsgg_a[:, i]
#         k = wsgg_k[:, i] * 100.0

#         i_b = Stefan_Boltzmann * (temperature_profiles ** 4) * a / torch.pi
#         i_b1 = Stefan_Boltzmann * (T_l ** 4) * a[0] / torch.pi
#         i_b2 = Stefan_Boltzmann * (T_r ** 4) * a[-1] / torch.pi

#         cur_G, cur_q, cur_dq = solve_rte(xs, k, i_b, i_b1, i_b2)

#         wsgg_G += cur_G
#         wsgg_q += cur_q
#         wsgg_dq += cur_dq

#     return wsgg_G, wsgg_q, wsgg_dq


# class EPSILON_LOSS(torch.nn.Module):
#     def __init__(self, lmbda=1 / 10.0):
#         super(EPSILON_LOSS, self).__init__()
#         self.loss_func = torch.nn.MSELoss()
#         self.lmbda = lmbda

#     def transform(self, x):
#         return (x ** self.lmbda - 1.0) / self.lmbda

#     def forward(self, weights, abscs, loss_additional, lbl_epsilons):
#         """
#         参数:
#             weights:         [B, 9]
#             abscs:           [B, 9]
#             loss_additional: [B, 4]
#                              [:, 0:3] -> [x_h2o, x_co2, x_co]
#                              [:, 3:4] -> L
#             lbl_epsilons:    [B, 1]
#         """
#         wsgg_epsilons = nn_wsgg_epsilon(
#             weights,
#             abscs,
#             loss_additional[:, 0:3],
#             loss_additional[:, 3:4]
#         ).reshape(-1, 1)

#         return self.loss_func(wsgg_epsilons, lbl_epsilons)
# import torch
# from scipy.constants import Stefan_Boltzmann
# from .rte import solve_rte
# from .mlp import MLP


# class NN_WSGG(torch.nn.Module):
#     """
#     输入:
#         x = [T, x_h2o, x_co2, x_co, p]

#     输出:
#         weights: 9个权重（8个灰气体 + 1个透明气体）
#         abscs:   9个吸收系数，其中第0个透明气体固定为0，其余8个由网络预测

#     说明:
#         - 权重受温度、组分、压强共同影响
#         - 吸收系数 abscs 也是等效吸收系数，已包含压强效应
#         - 因此后处理时不要再额外乘 pressure
#     """
#     def __init__(self, hidden_dims, x_mean: torch.Tensor, x_std: torch.Tensor):
#         super(NN_WSGG, self).__init__()

#         self.mlp_weights = MLP([5, *hidden_dims, 9])
#         self.mlp_abscs = MLP([4, *hidden_dims, 8])

#         self.register_buffer("x_mean", x_mean)
#         self.register_buffer("x_std", x_std)

#     def forward(self, x):
#         x = (x - self.x_mean) / self.x_std

#         weights = torch.softmax(self.mlp_weights(x), dim=1)

#         abscs = torch.zeros_like(weights)
#         abscs[:, 1:] = torch.exp(self.mlp_abscs(x[:, 1:]))

#         return weights, abscs


# def nn_wsgg_epsilon(weights, abscs, xs, L):
#     """
#     计算总发射率 epsilon

#     参数:
#         weights: [B, 9]
#         abscs:   [B, 9]，已包含压强效应的等效吸收系数
#         xs:      [B, 3] -> [x_h2o, x_co2, x_co]
#         L:       [B, 1] -> 光程长度

#     说明:
#         abscs 已经包含压强效应，
#         因此这里不能再使用 pL，只使用 L
#     """
#     sum_xs = torch.sum(xs, dim=1, keepdim=True)
#     epsilon = torch.sum(weights * (1.0 - torch.exp(-abscs * sum_xs * L)), dim=1)
#     return epsilon


# def nn_wsgg_rte(
#     nn_wsgg,
#     xs,
#     temperature_profiles,
#     x_h2o_profiles,
#     x_co2_profiles,
#     x_co_profiles,
#     pressure_profiles,
#     T_l,
#     T_r
# ):
#     nn_input = torch.vstack([
#         temperature_profiles,
#         x_h2o_profiles,
#         x_co2_profiles,
#         x_co_profiles,
#         pressure_profiles
#     ]).T

#     wsgg_a, wsgg_k = nn_wsgg(nn_input)

#     # 这里只乘组分和；不要再乘 pressure_profiles
#     gas_sum = (x_h2o_profiles + x_co2_profiles + x_co_profiles).reshape(-1, 1)
#     wsgg_k = wsgg_k * gas_sum

#     wsgg_G = 0.0
#     wsgg_q = 0.0
#     wsgg_dq = 0.0

#     for i in range(9):
#         a = wsgg_a[:, i]
#         k = wsgg_k[:, i] * 100.0

#         i_b = Stefan_Boltzmann * (temperature_profiles ** 4) * a / torch.pi
#         i_b1 = Stefan_Boltzmann * (T_l ** 4) * a[0] / torch.pi
#         i_b2 = Stefan_Boltzmann * (T_r ** 4) * a[-1] / torch.pi

#         cur_G, cur_q, cur_dq = solve_rte(xs, k, i_b, i_b1, i_b2)

#         wsgg_G += cur_G
#         wsgg_q += cur_q
#         wsgg_dq += cur_dq

#     return wsgg_G, wsgg_q, wsgg_dq


# class EPSILON_LOSS(torch.nn.Module):
#     def __init__(self, lmbda=1 / 10.0):
#         super(EPSILON_LOSS, self).__init__()
#         self.loss_func = torch.nn.MSELoss()
#         self.lmbda = lmbda

#     def transform(self, x):
#         return (x ** self.lmbda - 1.0) / self.lmbda

#     def forward(self, weights, abscs, loss_additional, lbl_epsilons):
#         """
#         参数:
#             weights:         [B, 9]
#             abscs:           [B, 9]
#             loss_additional: [B, 4]
#                              [:, 0:3] -> [x_h2o, x_co2, x_co]
#                              [:, 3:4] -> L
#             lbl_epsilons:    [B, 1]
#         """
#         wsgg_epsilons = nn_wsgg_epsilon(
#             weights,
#             abscs,
#             loss_additional[:, 0:3],
#             loss_additional[:, 3:4]
#         ).reshape(-1, 1)

#         return self.loss_func(wsgg_epsilons, lbl_epsilons)
# import torch
# from scipy.constants import Stefan_Boltzmann
# from .rte import solve_rte
# from .mlp import MLP


# class NN_WSGG(torch.nn.Module):
#     """
#     输入:
#         x = [T, x_h2o, x_co2, x_co, p]

#     输出:
#         weights: 9个权重（8个灰气体 + 1个透明气体）
#         abscs:   9个吸收系数，其中第0个透明气体固定为0，其余8个由网络预测

#     说明:
#         - 权重受温度、组分、压强共同影响
#         - 吸收系数 abscs 也是等效吸收系数，已包含压强效应
#         - 因此后处理时不要再额外乘 pressure
#     """
#     def __init__(self, hidden_dims, x_mean: torch.Tensor, x_std: torch.Tensor):
#         super(NN_WSGG, self).__init__()

#         # 输入: [T, x_h2o, x_co2, x_co, p]
#         self.mlp_weights = MLP([5, *hidden_dims, 9])

#         # 输入: [x_h2o, x_co2, x_co, p]
#         self.mlp_abscs = MLP([4, *hidden_dims, 8])

#         self.register_buffer("x_mean", x_mean)
#         self.register_buffer("x_std", x_std)

#     def forward(self, x):
#         """
#         参数:
#             x: shape [N, 5]
#                每列依次为 [T, x_h2o, x_co2, x_co, p]
#         返回:
#             weights: shape [N, 9]
#             abscs:   shape [N, 9]
#         """
#         x = (x - self.x_mean) / self.x_std

#         # 权重由 [T, x_h2o, x_co2, x_co, p] 决定
#         weights = torch.softmax(self.mlp_weights(x), dim=1)

#         # 吸收系数由 [x_h2o, x_co2, x_co, p] 决定
#         # 第0个为透明气体，其吸收系数固定为0
#         abscs = torch.zeros_like(weights)
#         abscs[:, 1:] = torch.exp(self.mlp_abscs(x[:, 1:]))

#         return weights, abscs


# def nn_wsgg_epsilon(weights, abscs, xs, pL):
#     """
#     计算总发射率 epsilon

#     参数:
#         weights: [B, 9]
#         abscs:   [B, 9]，已包含压强效应的等效吸收系数
#         xs:      [B, 3] -> [x_h2o, x_co2, x_co]
#         pL:      [B, 1] -> 压强-光程乘积

#     说明:
#         这里仍然保留 pL，因为你的训练标签/公式本身就是 epsilon(pL)
#         abscs 是等效吸收系数，不需要再额外乘 pressure
#     """
#     sum_xs = torch.sum(xs, dim=1, keepdim=True)
#     epsilon = torch.sum(weights * (1.0 - torch.exp(-abscs * sum_xs * pL)), dim=1)
#     return epsilon


# def nn_wsgg_rte(
#     nn_wsgg,
#     xs,
#     temperature_profiles,
#     x_h2o_profiles,
#     x_co2_profiles,
#     x_co_profiles,
#     pressure_profiles,
#     T_l,
#     T_r
# ):
#     """
#     基于 NN-WSGG 求解 RTE

#     参数:
#         nn_wsgg:              网络模型
#         xs:                   空间坐标
#         temperature_profiles: [N]
#         x_h2o_profiles:       [N]
#         x_co2_profiles:       [N]
#         x_co_profiles:        [N]
#         pressure_profiles:    [N]
#         T_l:                  左边界温度
#         T_r:                  右边界温度

#     说明:
#         - pressure_profiles 作为网络输入，影响 weights 和 abscs
#         - abscs 已包含压强效应，因此这里不要再乘 pressure_profiles
#         - 这里只保留组分和对吸收系数的缩放
#     """
#     nn_input = torch.vstack([
#         temperature_profiles,
#         x_h2o_profiles,
#         x_co2_profiles,
#         x_co_profiles,
#         pressure_profiles
#     ]).T

#     wsgg_a, wsgg_k = nn_wsgg(nn_input)

#     # 只乘参与辐射气体的组分和
#     # 不再额外乘 pressure_profiles，因为 wsgg_k 已经包含压强效应
#     gas_sum = (x_h2o_profiles + x_co2_profiles + x_co_profiles).reshape(-1, 1)
#     wsgg_k = wsgg_k * gas_sum

#     wsgg_G = 0.0
#     wsgg_q = 0.0
#     wsgg_dq = 0.0

#     for i in range(9):
#         a = wsgg_a[:, i]
#         k = wsgg_k[:, i] * 100.0

#         i_b = Stefan_Boltzmann * (temperature_profiles ** 4) * a / torch.pi
#         i_b1 = Stefan_Boltzmann * (T_l ** 4) * a[0] / torch.pi
#         i_b2 = Stefan_Boltzmann * (T_r ** 4) * a[-1] / torch.pi

#         cur_G, cur_q, cur_dq = solve_rte(xs, k, i_b, i_b1, i_b2)

#         wsgg_G += cur_G
#         wsgg_q += cur_q
#         wsgg_dq += cur_dq

#     return wsgg_G, wsgg_q, wsgg_dq


# class EPSILON_LOSS(torch.nn.Module):
#     def __init__(self, lmbda=1 / 10.0):
#         super(EPSILON_LOSS, self).__init__()
#         self.loss_func = torch.nn.MSELoss()
#         self.lmbda = lmbda

#     def transform(self, x):
#         return (x ** self.lmbda - 1.0) / self.lmbda

#     def forward(self, weights, abscs, loss_additional, lbl_epsilons):
#         """
#         参数:
#             weights:         [B, 9]
#             abscs:           [B, 9]
#             loss_additional: [B, 4]
#                              [:, 0:3] -> [x_h2o, x_co2, x_co]
#                              [:, 3:4] -> pL
#             lbl_epsilons:    [B, 1]
#         """
#         wsgg_epsilons = nn_wsgg_epsilon(
#             weights,
#             abscs,
#             loss_additional[:, 0:3],
#             loss_additional[:, 3:4]
#         ).reshape(-1, 1)

#         return self.loss_func(wsgg_epsilons, lbl_epsilons)
    
# # allP v1
# # import torch
# # from scipy.constants import Stefan_Boltzmann
# # from .rte import solve_rte
# # from .mlp import MLP


# # class NN_WSGG(torch.nn.Module):
# #     """
# #     输入:
# #         x = [T, x_h2o, x_co2, x_co, p]

# #     输出:
# #         weights: 9个权重（8个灰气体 + 1个透明气体）
# #         abscs:   9个吸收系数，其中第0个透明气体固定为0，其余8个由网络预测

# #     说明:
# #         - 权重受温度、组分、压强共同影响
# #         - 吸收系数 abscs 是等效吸收系数，已包含压强效应
# #         - 因此后处理时不要再额外乘 pressure
# #         - 压力输入在 forward 内部先做 log 变换，再参与 z-score 归一化
# #           （x_mean[4] 和 x_std[4] 必须是基于 log(p) 计算的统计量）
# #     """
# #     def __init__(self, hidden_dims, x_mean: torch.Tensor, x_std: torch.Tensor):
# #         super(NN_WSGG, self).__init__()

# #         # 输入: [T, x_h2o, x_co2, x_co, p]
# #         self.mlp_weights = MLP([5, *hidden_dims, 9])

# #         # 输入: [x_h2o, x_co2, x_co, p（log 变换后归一化）]
# #         self.mlp_abscs = MLP([4, *hidden_dims, 8])

# #         self.register_buffer("x_mean", x_mean)
# #         self.register_buffer("x_std", x_std)

# #     def forward(self, x):
# #         """
# #         参数:
# #             x: shape [N, 5]
# #                每列依次为 [T, x_h2o, x_co2, x_co, p]
# #                p 为原始压力值（atm / bar，正数即可），内部自动做 log 变换
# #         返回:
# #             weights: shape [N, 9]
# #             abscs:   shape [N, 9]
# #         """
# #         # -------------------------------------------------------
# #         # 关键修改：对压力列（第 4 列）先做 log 变换，再 z-score
# #         # 原始 p 范围 0.1~80 bar，线性差 800x；log 空间约 -2.3~4.4，
# #         # z-score 后与其他特征尺度一致，网络可以均匀"看见"所有压力。
# #         # 注意：x_mean[4] / x_std[4] 必须是对 log(p) 计算的统计量。
# #         # -------------------------------------------------------
# #         x = x.clone()
# #         x[:, 4] = torch.log(x[:, 4].clamp_min(1e-6))   # log(p)

# #         x_norm = (x - self.x_mean) / self.x_std         # z-score（含 log p）

# #         # 权重由 [T, x_h2o, x_co2, x_co, log(p)] 决定
# #         weights = torch.softmax(self.mlp_weights(x_norm), dim=1)

# #         # 吸收系数由 [x_h2o, x_co2, x_co, log(p)] 决定
# #         # 第0个为透明气体，吸收系数固定为 0
# #         abscs = torch.zeros_like(weights)
# #         abscs[:, 1:] = torch.exp(self.mlp_abscs(x_norm[:, 1:]))

# #         return weights, abscs


# # def nn_wsgg_epsilon(weights, abscs, xs, pL):
# #     """
# #     计算总发射率 epsilon

# #     参数:
# #         weights: [B, 9]
# #         abscs:   [B, 9]，已包含压强效应的等效吸收系数
# #         xs:      [B, 3] -> [x_h2o, x_co2, x_co]
# #         pL:      [B, 1] -> 压强-光程乘积

# #     说明:
# #         abscs 是等效吸收系数，不需要再额外乘 pressure
# #     """
# #     sum_xs = torch.sum(xs, dim=1, keepdim=True)
# #     epsilon = torch.sum(weights * (1.0 - torch.exp(-abscs * sum_xs * pL)), dim=1)
# #     return epsilon


# # def nn_wsgg_rte(
# #     nn_wsgg,
# #     xs,
# #     temperature_profiles,
# #     x_h2o_profiles,
# #     x_co2_profiles,
# #     x_co_profiles,
# #     pressure_profiles,
# #     T_l,
# #     T_r
# # ):
# #     """
# #     基于 NN-WSGG 求解 RTE

# #     参数:
# #         nn_wsgg:              网络模型
# #         xs:                   空间坐标
# #         temperature_profiles: [N]
# #         x_h2o_profiles:       [N]
# #         x_co2_profiles:       [N]
# #         x_co_profiles:        [N]
# #         pressure_profiles:    [N]  ← 传入原始压力值即可，forward 内部做 log 变换
# #         T_l:                  左边界温度
# #         T_r:                  右边界温度
# #     """
# #     nn_input = torch.vstack([
# #         temperature_profiles,
# #         x_h2o_profiles,
# #         x_co2_profiles,
# #         x_co_profiles,
# #         pressure_profiles          # 原始 p，forward 内部自动 log 变换
# #     ]).T

# #     wsgg_a, wsgg_k = nn_wsgg(nn_input)

# #     # 只乘参与辐射气体的组分和；wsgg_k 已包含压强效应，不再额外乘 pressure
# #     gas_sum = (x_h2o_profiles + x_co2_profiles + x_co_profiles).reshape(-1, 1)
# #     wsgg_k = wsgg_k * gas_sum

# #     wsgg_G  = 0.0
# #     wsgg_q  = 0.0
# #     wsgg_dq = 0.0

# #     for i in range(9):
# #         a = wsgg_a[:, i]
# #         k = wsgg_k[:, i] * 100.0

# #         i_b  = Stefan_Boltzmann * (temperature_profiles ** 4) * a / torch.pi
# #         i_b1 = Stefan_Boltzmann * (T_l ** 4) * a[0]  / torch.pi
# #         i_b2 = Stefan_Boltzmann * (T_r ** 4) * a[-1] / torch.pi

# #         cur_G, cur_q, cur_dq = solve_rte(xs, k, i_b, i_b1, i_b2)

# #         wsgg_G  += cur_G
# #         wsgg_q  += cur_q
# #         wsgg_dq += cur_dq

# #     return wsgg_G, wsgg_q, wsgg_dq


# # class EPSILON_LOSS(torch.nn.Module):
# #     def __init__(self, lmbda=1 / 10.0):
# #         super(EPSILON_LOSS, self).__init__()
# #         self.loss_func = torch.nn.MSELoss()
# #         self.lmbda = lmbda

# #     def transform(self, x):
# #         return (x ** self.lmbda - 1.0) / self.lmbda

# #     def forward(self, weights, abscs, loss_additional, lbl_epsilons):
# #         """
# #         参数:
# #             weights:         [B, 9]
# #             abscs:           [B, 9]
# #             loss_additional: [B, 4]
# #                              [:, 0:3] -> [x_h2o, x_co2, x_co]
# #                              [:, 3:4] -> pL
# #             lbl_epsilons:    [B, 1]
# #         """
# #         wsgg_epsilons = nn_wsgg_epsilon(
# #             weights,
# #             abscs,
# #             loss_additional[:, 0:3],
# #             loss_additional[:, 3:4]
# #         ).reshape(-1, 1)

# #         return self.loss_func(wsgg_epsilons, lbl_epsilons)
