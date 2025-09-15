print('hi')
import os
from tqdm import tqdm
from src import *
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else 'cpu'

batch_size = 2
epochs = 2
lr = 3e-3

features = 32
context_features = 5
image_size = (16, 16)
save_dir = 'weights/'

T = 500
beta_start = 1e-3
beta_end = 0.02
betas = (beta_end - beta_start) * torch.linspace(0, 1, T+1, device=device) + beta_start
alphas = 1 - betas
alphas_hat = torch.cumsum(alphas.log(), dim=0).exp() # numerical stability trick
alphas_hat[0] = 1

# DATASET
dataset = SpritesDataset("data/sprites/sprites_1788_16x16.npy",
                        "data/sprites/sprite_labels_nc_1788_16x16.npy",
                        sprites_transform,
                        null_context=False)
model = ContextUnet(in_channels=3, features=features, context_features=context_features, image_size=image_size).to(device)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1) if device=='cuda' else DataLoader(dataset, batch_size, shuffle=True)
optim = torch.optim.Adam(model.parameters(), lr=lr)
model.train()

# TRAIN 
def perturb_input(x, t, noise):
    return alphas_hat.sqrt()[t, None, None, None]*x + (1-alphas_hat[t, None, None, None])*noise
losses_save = []
for epoch in range(epochs):
    optim.param_groups[0]['lr'] = lr*(1-epoch/epochs)

    losses = 0
    for x, c in dataloader:
        optim.zero_grad()
        x = x.to(device)
        c = c.to(device)

        context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(device)
        c = c * context_mask.unsqueeze(-1)

        noise = torch.randn_like(x)
        t = torch.randint(1, T+1, (x.shape[0],)).to(device)
        x_pert = perturb_input(x, t, noise)
        
        pred_noise = model(x_pert, t/T, c)

        loss = F.mse_loss(pred_noise, noise)
        losses += (loss.item())
        loss.backward()

        optim.step()
    print(f'epoch: {epoch} - loss: {np.mean(losses)}')
    losses_save.append(np.mean(losses))
    
    # save model periodically
    if epoch%5==0 or epoch==int(epochs-1):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(model.state_dict(), save_dir + f"model_{epoch}.pth")
        print('saved model at ' + save_dir + f"model_{epoch}.pth")
print('saved model at ' + save_dir + f"model_{epoch}.pth")

def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = betas.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - alphas[t]) / (1 - alphas_hat[t]).sqrt())) / alphas[t].sqrt()
    return mean + noise
@torch.no_grad()
def sample_ddpm(n_sample, context ,save_rate=20):
    samples = torch.randn(n_sample, 3, image_size[0], image_size[1]).to(device)

    intermediate = []
    for i in range(T, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')
        t = torch.tensor([i / T])[:, None, None, None].to(device)
        z = torch.randn_like(samples) if i > 1 else 0
        eps = model(samples, t, c=context)
        samples = denoise_add_noise(samples, i, eps, z)
        if i % save_rate ==0 or i==T or i<8:
            intermediate.append(samples.detach().cpu().numpy())
        
    intermediate = np.stack(intermediate)
    return samples, intermediate

def denoise_ddim(x, t, t_prev, pred_noise):
    ab = alphas_hat[t]
    ab_prev = alphas_hat[t_prev]

    x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - (1 - ab).sqrt() * pred_noise)
    dir_xt = (1 - ab_prev).sqrt() * pred_noise
    return x0_pred + dir_xt
@torch.no_grad()
def sample_ddim(n_sample, context ,n=20):
    samples = torch.randn(n_sample, 3, image_size[0], image_size[1]).to(device)

    intermediate = []
    step_size = T // n
    for i in range(T, 0, -step_size):
        print(f'sampling timestep {i:3d}', end='\r')

        t = torch.tensor([i / T])[:, None, None, None].to(device)
        eps = model(samples, t, c=context)
        samples = denoise_ddim(samples, i, i-step_size, eps)
        intermediate.append(samples.detach().cpu().numpy())
        
    intermediate = np.stack(intermediate)
    return samples, intermediate

model.load_state_dict(torch.load(f"{save_dir}/model_{str(epoch)}.pth", map_location=device))
model.eval()
print("Loaded in Model")

# visualize samples
ctx = torch.tensor([
    # hero, non-hero, food, spell, side-facing
    [1,0,0,0,0],  
    [1,0,0,0,0],    
    [0,0,0,0,1],
    [0,0,0,0,1],    
    [0,1,0,0,0],
    [0,1,0,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
]).float().to(device)

import time
start = time.time()
samples, _ = sample_ddpm(ctx.shape[0], ctx)
speed = time.time()-start
print(f'DDPM: generated outputs. taken {speed} time')
torch.save(samples, './outs/ddpm_file.pt')
torch.save(torch.tensor(losses_save), './outs/ddpm_loss.pt')

start = time.time()
samples, _ = sample_ddim(ctx.shape[0], context=ctx, n=25)
speed = time.time()-start
print(f'DDIM: generated outputs. taken {speed} time')
torch.save(samples, './outs/ddim_file.pt')
torch.save(torch.tensor(losses_save), './outs/ddim_loss.pt')