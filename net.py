import torch

from util import *
import gc
from efficient_kan import KAN


class LambdaLayer(nn.Module):
    def __init__(self, lambda_function):
        super(LambdaLayer, self).__init__()
        self.lambda_function = lambda_function

    def forward(self, x):
        return self.lambda_function(x)


class VrfOptimizer(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(VrfOptimizer, self).__init__()

        self.layer = nn.Sequential(
            # KAN([input_size, hidden_size, output_size]),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, gradient):

        gradient = gradient.unsqueeze(0)
        gradient = self.layer(gradient)
        gradient = gradient.squeeze(0)
        return gradient


class VdOptimizer(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(VdOptimizer, self).__init__()

        self.layer = nn.Sequential(
            # KAN([input_size, hidden_size, output_size]),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, gradient):

        gradient = gradient.unsqueeze(0)
        gradient = self.layer(gradient)
        gradient = gradient.squeeze(0)
        return gradient


def NativeVdGradientbasedLearner(optimizee, Internal_iteration, user_weights, Channel, Vd,
                    Vrf, noise_power, retain_graph_flag=True):
    L_vd = 0
    Vd_internal = Vd  # initialize the compressed precoding matrix
    Vd_internal.requires_grad = True  # set the requires_grad flag to true to enable the backward propagation
    sum_loss_vd = 0  # record the accumulated loss
    for internal_index in range(Internal_iteration):
        SE, L_vd = compute_weighted_sum_rate(Channel, Vd_internal, Vrf, user_weights, noise_power)
        sum_loss_vd = L_vd + sum_loss_vd  # accumulate the loss
        sum_loss_vd.backward(retain_graph=retain_graph_flag)  # compute the gradient
        Vd_grad = Vd_internal.grad.clone().detach()  # clone the gradient
        #  as pytorch can not process complex number, we have to split the real and imaginary parts and concatenate them
        Vd_grad1 = torch.cat((Vd_grad.real, Vd_grad.imag), dim=1)  # concatenate the real and imaginary part
        shape = Vd_grad1.shape
        Vd_grad1 = Vd_grad1.reshape(-1)
        Vd_update = optimizee(Vd_grad1)  # input the gradient and get the increment
        Vd_update = Vd_update.reshape(shape)
        Vd_update1 = Vd_update[:, 0: nr_of_users] + 1j * Vd_update[:, nr_of_users: 2 * nr_of_users]
        Vd_internal = Vd_internal + Vd_update1  # update the compressed precoding matrix

        Vd_update.retain_grad()
        Vd_internal.retain_grad()

    return L_vd, sum_loss_vd, Vd_internal



def NativeVrfGradientbasedLearner(optimizee, Internal_iteration, user_weights, Channel, Vd,
                     Vrf, noise_power, retain_graph_flag=True):
    L_vrf = 0
    vrf_internal = Vrf
    vrf_internal.requires_grad = True
    sum_loss_vrf = 0
    for internal_index in range(Internal_iteration):
        SE, L_vrf = compute_weighted_sum_rate(Channel, Vd, vrf_internal, user_weights, noise_power)
        sum_loss_vrf = L_vrf + sum_loss_vrf
        sum_loss_vrf.backward(retain_graph=retain_graph_flag)
        Vrf_grad = vrf_internal.grad.clone().detach()
        Vrf_grad1 = torch.cat((Vrf_grad.real, Vrf_grad.imag), dim=1)
        shape = Vrf_grad1.shape
        Vrf_grad1 = Vrf_grad1.reshape(-1)
        vrf_update = optimizee(Vrf_grad1)
        vrf_update = vrf_update.reshape(shape)
        vrf_update1 = vrf_update[:, 0: nr_of_rfs] + 1j * vrf_update[:, nr_of_rfs: 2 * nr_of_rfs]
        vrf_riemannian_grad = euclidean_orthogonal_projection(vrf_update1, vrf_internal)
        vrf_internal = retraction(vrf_internal, vrf_update1)
        vrf_update.retain_grad()
        vrf_internal.retain_grad()
    return L_vrf, sum_loss_vrf, vrf_internal


input_size_vd = nr_of_rfs * nr_of_users * 2
hidden_size_vd = 200
output_size_vd = nr_of_rfs * nr_of_users * 2

input_size_vrf = nr_of_BS_antennas * nr_of_rfs * 2
hidden_size_vrf = 200
output_size_vrf = nr_of_BS_antennas * nr_of_rfs * 2


