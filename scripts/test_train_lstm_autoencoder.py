# %%
import sys
import pathlib
# Make "src" importable without installing the package yet
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pprint import pprint
from tqdm.auto import tqdm  

# --- load and preprocess ---

# from planetary_systems.dataset import Dataset
from planetary_systems.dataset_planetary_systems_with_existance import DatasetPlanetarySystemsWithExistance

data = DatasetPlanetarySystemsWithExistance(
    pathlib.Path(__file__).parent.parent / "data" / "Easier Dataset.csv"
)

dataloader = data.dataloader
dataset = data.dataset
padded = data.padded

data.normalized.hist()
plt.tight_layout()
plt.show()

data.padded.hist()
plt.tight_layout()
plt.show()

# %%
columns = ["a","total_mass","r","exists","does_not_exist"]

num_planets= data.num_planets
input_dim = len(columns)
embed_dim = 35
hidden_dim = 200
num_layers = 2
dropout = 0.

pprint({
    "num_planets": num_planets,
    "input_dim": input_dim,
    "embed_dim": embed_dim,
    "hidden_dim": hidden_dim,
    "num_layers": num_layers,
    "dropout": dropout
})

# %%

from planetary_systems.models.lstm_autoencoder import SystemAutoencoder, HybridLoss


# --- training ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SystemAutoencoder().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = HybridLoss(alpha=10.0)

losses = []

# %%
for epoch in tqdm(range(10)):
    for xb, _ in dataloader:
        xb = xb.to(device)
        opt.zero_grad()
        recon, z, mu, logvar = model(xb)
        loss = loss_fn(recon, xb, mu, logvar)
        loss.backward()
        opt.step()
    losses.append(loss.item())
    print(f"Epoch {epoch:02d} | Loss {loss.item():.4f}")
    
    if epoch % 5 != 0:
        continue
    
    # --- extract embeddings ---
    embeddings = []
    with torch.no_grad():
        for xb, _ in dataloader:
            xb = xb.to(device)
            _, z, mu, logvar = model(xb)
            embeddings.append(z.cpu())
    embeddings = torch.cat(embeddings).numpy()

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction loss")
    plt.title("Training loss curve")
    
    pca = PCA(n_components=2)
    proj = pca.fit_transform(embeddings)
    plt.subplot(1,2,2)
    plt.scatter(proj[:,0], proj[:,1], s=3, alpha=0.5)
    plt.title("Embeddings (PCA to 2D)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.axis("equal")
    plt.grid()
    plt.show()

# %%
model.eval()
for i in range(20):
    x, n_planets = dataset[i]
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        recon, z, mu, logvar = model(x)
    plt.figure()
 
    plt.subplot(311)
    plt.title(f"System {i} (feature=a, {n_planets} planets)")
    orig = x[0,:n_planets,0].cpu().numpy()
    rec = recon[0,:n_planets,0].cpu().numpy()
    plt.plot(orig, "o-", label="orig")
    plt.plot(rec, "x--", label="recon")
    
    plt.subplot(312)
    plt.title(f"System {i} (feature=m, {n_planets} planets)")
    orig = x[0,:n_planets,1].cpu().numpy()
    rec = recon[0,:n_planets,1].cpu().numpy()
    plt.plot(orig, "o-", label="orig")
    plt.plot(rec, "x--", label="recon")

    plt.subplot(313)
    plt.title(f"System {i} (feature=r, {n_planets} planets)")
    orig = x[0,:n_planets,2].cpu().numpy()
    rec = recon[0,:n_planets,2].cpu().numpy()
    plt.plot(orig, "o-", label="orig")
    plt.plot(rec, "x--", label="recon")

    plt.legend()
    plt.show()

# %%
# Sample from latent space and decode
for i in range(5):
    with torch.no_grad():
        mean = torch.tensor(embeddings.mean(0), dtype=torch.float32).to(device)
        std = torch.tensor(embeddings.std(0), dtype=torch.float32).to(device)
        z_rand = torch.randn(1, embed_dim).to(device) * std + mean
        recon = model.decode(z_rand, num_planets)
        recon = recon.squeeze(0).cpu().numpy()
    
    r = data.normalizer.inverse_transform(
        pd.DataFrame(recon, columns=columns)
    )
    r = r[r["exists"] > r["does_not_exist"]]
    plt.plot(r["a"], "o-")
    plt.show()
    
# %% 