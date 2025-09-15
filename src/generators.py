def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = betas.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - alphas[t]) / (1 - alphas_hat[t]).sqrt())) / alphas[t].sqrt()
    return mean + noise

