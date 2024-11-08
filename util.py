
import math
import numpy as np
import torch
import torch.nn as nn
import random

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
External_iteration = 500
Internal_iteration = 1
Update_steps = 1
N_i = Internal_iteration
N_o = Update_steps

optimizer_lr_vd = 10e-4
optimizer_lr_vrf = 15e-4

nr_of_users = 4
nr_of_BS_antennas = 64
nr_of_rfs = 4
nr_of_clusters = 3
nr_of_rays = 30

epoch = 1
nr_of_training = 50
snr = 10
total_power = 1
noise_power = total_power / 10**(snr/10)



def mmWave(nr_of_BS_antennas, nr_of_clusters, nr_of_rays, nr_of_users):

    # 角度扩展设置
    sigma_aod = 10 * torch.pi / 180  # AOD 的标准差（弧度）
    b_aod = sigma_aod / torch.sqrt(torch.tensor(2.0))  # AOD 的尺度参数

    # 初始化信道矩阵和阵列响应矩阵
    Hk = torch.zeros(nr_of_users, nr_of_BS_antennas, dtype=torch.cfloat)  # Nr=1, 表示 MISO
    AT = torch.zeros(nr_of_BS_antennas, nr_of_clusters * nr_of_rays, dtype=torch.cfloat)

    for k in range(nr_of_users):
        # 为每个聚簇随机生成 AOD 均值
        E_aod = 2 * torch.pi * torch.rand(nr_of_clusters)

        # 在 (-0.5, 0.5) 范围内均匀分布的随机数
        a_aod = torch.rand(nr_of_clusters, nr_of_rays) - 0.5

        # 拉普拉斯分布的 AOD
        aod = E_aod.unsqueeze(1).repeat(1, nr_of_rays) - b_aod * torch.sign(a_aod) * torch.log(1 - 2 * torch.abs(a_aod))
        aod = torch.sin(aod)

        # 计算信号的阵列响应
        signature_t = torch.arange(nr_of_BS_antennas).unsqueeze(1) * 1j * torch.pi

        # 初始化射线信道矩阵
        H_ray = torch.zeros(nr_of_BS_antennas, nr_of_clusters, nr_of_rays, dtype=torch.cfloat)

        for i in range(nr_of_clusters):
            for m in range(nr_of_rays):
                H_ray[:, i, m] = (torch.randn(1) + 1j * torch.randn(1)) / torch.sqrt(torch.tensor(2)) * torch.exp(aod[i, m] * signature_t).T.conj() / torch.sqrt(torch.tensor(nr_of_BS_antennas))

        # 对每个聚簇的射线进行求和
        H_cl = torch.sum(H_ray, dim=2)

        # 对所有聚簇进行求和并归一化
        Hk[k, :] = torch.sqrt(torch.tensor(nr_of_BS_antennas / (nr_of_clusters * nr_of_rays), dtype=torch.float)) * torch.sum(H_cl, dim=1)

    return Hk


def compute_weighted_sum_rate(H, Vd, Vrf, user_weights, noise_power):
    # 确保所有输入张量都在 GPU 上

    rate = torch.zeros(nr_of_users, dtype=torch.float64)
    R = torch.zeros(nr_of_users)
    for i in range(nr_of_users):
        denominator = torch.tensor(0, dtype=torch.float64)

        for l in range(nr_of_users):
            # 确保所有的矩阵乘法都在 GPU 上执行
            product = torch.matmul(H[i, :], torch.matmul(Vrf, Vd[:, l]))
            denominator += product.abs() ** 2

        numerator = torch.matmul(H[i, :], torch.matmul(Vrf, Vd[:, i])).abs() ** 2
        denominator = denominator - numerator + noise_power

        rate[i] = torch.log2(1 + numerator / denominator)
        R[i] = user_weights[i] * rate[i]

    system_rate = torch.sum(R)
    loss_fun = -system_rate + 0.2 * torch.std(rate) ** 2

    return system_rate, loss_fun


def init_Vd(total_power, Vrf):

    transmitter_precoder = torch.randn(nr_of_rfs, nr_of_users) + 1j * torch.randn(nr_of_rfs, nr_of_users)
    normV = torch.norm(Vrf @ transmitter_precoder)  # compute the norm of the precoding matrix before normalization
    WW = math.sqrt(total_power) / normV
    transmitter_precoder_init = transmitter_precoder * WW

    return transmitter_precoder_init


def euclidean_orthogonal_projection(Vrf_grad, Vrf):

    return Vrf_grad - ((Vrf_grad * Vrf.conj()).real) * Vrf


def retraction(Vrf, vrf_riemannian_grad):

    return (Vrf + vrf_riemannian_grad) / torch.abs(Vrf + vrf_riemannian_grad)


def seed_everything(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)  # cpu random seed
    np.random.seed(seed)  # numpy random seed
    random.seed(seed)  # python random seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
