import torch
def drop_out(x,drop_prob):
    drop_prob=torch.rand(1) * drop_prob
    mask=torch.rand_like(x)>drop_prob
    dropped_x=x*mask.float()

    return dropped_x