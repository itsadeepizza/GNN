import torch

def benchmark_nomove(position_with_noise, labels):
    "simulate not moving particles and calculate the loss"
    second_last_position = position_with_noise[:, -1, :].detach()
    loss_nomove = torch.abs(second_last_position - labels.detach()).sum()
    return loss_nomove

def benchmark_noacc(position_with_noise, labels):
    "simulate same speed particles and calculate the loss"
    last_velocity = position_with_noise[:, -1, :].detach() - position_with_noise[:, -2, :].detach()
    next_position = position_with_noise[:, -1, :].detach() + last_velocity
    loss_noacc = torch.abs(next_position - labels.detach()).sum()
    return loss_noacc

def benchmark_nojerk(position_with_noise, labels):
    "simulate same acceleration particles and calculate the loss"
    second_last_velocity = position_with_noise[:, -1, :].detach() - position_with_noise[:, -2, :].detach()
    third_last_velocity = position_with_noise[:, -2, :].detach() - position_with_noise[:, -3, :].detach()
    last_acc = second_last_acc = second_last_velocity - third_last_velocity
    last_velocity = second_last_velocity + last_acc

    next_position = position_with_noise[:, -1, :].detach() + last_velocity
    loss_nojerk = torch.abs(next_position - labels.detach()).sum()
    return loss_nojerk