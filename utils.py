import torch

def q_sample_target(x0, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    alpha_t = sqrt_alphas_cumprod[t]
    one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t]
    return alpha_t * x0 + one_minus_alpha_t * noise
