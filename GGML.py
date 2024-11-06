import random
import scipy.io as sio
import torch
import numpy as np
import math
import time
import torch.nn.init as init
from net import *


seed_everything(42)

# 加载数据
HH = sio.loadmat(f'dataset.mat')['HH']
user_weights = sio.loadmat(f'dataset.mat')['omega'].squeeze()
regulated_user_weights = torch.ones(nr_of_users) / nr_of_users

HH = torch.tensor(HH, dtype=torch.complex64)
SNR_dB = torch.arange(10, 11, 5)
# Delta = torch.arange(0, 0.41, 0.05)
for snr_index in range(len(SNR_dB)):
    snr = SNR_dB[snr_index]
    noise_power = total_power / 10 ** (snr / 10)
    # delta = Delta[snr_index]

    variable_name = f"WSR_List_GGML_Nt{nr_of_BS_antennas}_Nrf{nr_of_rfs}_K{nr_of_users}_SNR{snr}_1"
    file_name = f'WSR_List_GGML_Nt{nr_of_BS_antennas}_Nrf{nr_of_rfs}_K{nr_of_users}_SNR{snr}_1.mat'
    globals()[variable_name] = torch.zeros(nr_of_training)
    WSR_ImCSI = torch.zeros(nr_of_training)

    MLMO_run_time = 0

    temp = torch.zeros(nr_of_training)
    # 主训练循环
    for item_index in range(nr_of_training):
        # item_index += 25

        maxi = 0

        mm_Wave_Channel = HH[item_index, :, :].to(torch.complex64)
        # mm_Wave_Channel = mmWave(nr_of_BS_antennas, nr_of_clusters, nr_of_rays, nr_of_users)

        optimizer_vd = VdOptimizer(input_size_vd, hidden_size_vd, output_size_vd)
        adam_vd = torch.optim.Adam(optimizer_vd.parameters(), lr=optimizer_lr_vd)

        optimizer_vrf = VrfOptimizer(input_size_vrf, hidden_size_vrf, output_size_vrf)
        adam_vrf = torch.optim.Adam(optimizer_vrf.parameters(), lr=optimizer_lr_vrf)

        Vrf = torch.exp(1j * torch.rand(nr_of_BS_antennas, nr_of_rfs) * 2 * torch.pi)
        Vrf_init = Vrf.clone()
        # Vd = init_Vd(total_power, Vrf)
        H_temp = mm_Wave_Channel @ Vrf
        H_conj_transpose = H_temp.conj().T  # H 的共轭转置
        inv_HH = torch.inverse(torch.matmul(H_temp, H_conj_transpose))  # H*H' 的逆
        Vd = torch.matmul(H_conj_transpose, inv_HH)
        normV = torch.norm(Vrf @ Vd)
        Vd = Vd * math.sqrt(total_power) / normV
        Vd_init = Vd.clone()

        LossAccumulated_vd = 0
        LossAccumulated_vrf = 0
        start_time = time.time()

        for epoch in range(External_iteration):
            # 计算 meta learner 的损失
            _, _, Vrf = NativeVrfGradientbasedLearner(
                optimizer_vrf, Internal_iteration, regulated_user_weights, mm_Wave_Channel,
                Vd.clone().detach(), Vrf_init, noise_power
            )

            _, _, Vd = NativeVdGradientbasedLearner(
                optimizer_vd, Internal_iteration, regulated_user_weights,
                mm_Wave_Channel, Vd_init, Vrf.clone().detach(), noise_power
            )

            normV = torch.norm(Vrf @ Vd)
            Vd = Vd * math.sqrt(total_power) / normV

            # 计算 WSR 并累积损失
            WSR, loss_total = compute_weighted_sum_rate(mm_Wave_Channel, Vd, Vrf, regulated_user_weights, noise_power)
            LossAccumulated_vd += loss_total
            LossAccumulated_vrf += loss_total

            # globals()[variable_name][item_index] = WSR
            # 更新最大 WSR
            if WSR > maxi:
                maxi = WSR.item()
                Vd_opt = Vd
                Vrf_opt = Vrf
                # globals()[variable_name][item_index] = maxi


            # print(f"item {epoch} SE {WSR}")

            adam_vd.zero_grad()
            adam_vrf.zero_grad()

            # 计算平均损失并更新
            Average_loss_vd = LossAccumulated_vd / Update_steps
            Average_loss_vrf = LossAccumulated_vrf / Update_steps

            Average_loss_vd.backward(retain_graph=True)  # 去除 retain_graph=True
            Average_loss_vrf.backward(retain_graph=True)  # 去除 retain_graph=True

            adam_vd.step()
            adam_vrf.step()

            # 重置累积损失
            LossAccumulated_vd = 0
            LossAccumulated_vrf = 0

            # 记录运行时间
        MLMO_run_time = time.time() - start_time
        print("时间：", MLMO_run_time)
        gc.collect()
        torch.cuda.empty_cache()  # 清理缓存，避免内存泄漏
        print(f"item {item_index} 完成，最大值 {maxi}")

    print("均值：", torch.mean(WSR_ImCSI))
