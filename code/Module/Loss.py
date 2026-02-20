import torch
import torch.nn.functional as F


def global_reconstruction_loss(x_g, x_g_hat, sigma_g):
    loss = torch.sum(((x_g - x_g_hat) ** 2) / sigma_g + torch.log(sigma_g))
    return loss


def local_reconstruction_loss(x_l, x_l_hat, sigma_l):
    loss = torch.sum(((x_l - x_l_hat) ** 2) / sigma_l + torch.log(sigma_l))
    return loss


def expert_balance_loss(alpha_1, token_expert_map, expert_scores, num_experts, num_tokens):
    device = expert_scores.device

    token_expert_map = token_expert_map.to(device)
    expert_scores = expert_scores.to(device)

    f_i = torch.zeros(num_experts, device=device)
    P_i = torch.zeros(num_experts, device=device)

    for i in range(num_experts):
        f_i[i] = torch.sum(token_expert_map == i) / num_tokens
        P_i[i] = torch.sum(expert_scores[:, i]) / num_tokens

    loss = alpha_1 * torch.sum(f_i * P_i)
    return loss
