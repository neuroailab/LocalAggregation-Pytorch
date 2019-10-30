import torch


def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))


def repeat_1d_tensor(t, num_reps):
    return t.unsqueeze(1).expand(-1, num_reps)

