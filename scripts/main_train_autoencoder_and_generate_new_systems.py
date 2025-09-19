# %%
import pathlib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pprint import pprint
from tqdm.auto import tqdm
from sklearn.mixture import GaussianMixture

# --- load and preprocess ---
from planetary_systems.dataset_planetary_systems import DatasetPlanetarySystems

dataset = DatasetPlanetarySystems(pathlib.Path(__file__).parent.parent / "data" / "Easier Dataset.csv")

# --- collate: pad to max length in batch, return (x, lens) ---
def pad_collate(batch):
    lens = torch.tensor([b.shape[0] for b in batch], dtype=torch.long)
    T = int(lens.max().item())
    x = torch.zeros(len(batch), T, 3, dtype=torch.float32)
    for i, seq in enumerate(batch):
        x[i, :seq.shape[0]] = seq
    return x, lens

dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=pad_collate)


# %%  set up the model

if False:
    # use a transformer-encoded GRUVAE

    from planetary_systems.models.transformer_encoded_gruvae import TransEncGRUVAE
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    model = TransEncGRUVAE().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)

else:
    # use a standard GRUVAE

    from planetary_systems.models.gruvae import GRUVAE


    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    model = GRUVAE().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)


# %%  --- train ---


def loss_batch(x, x_hat, lens, mu, lv, beta=0.15):
    B, T, _ = x.shape
    mask = (torch.arange(T, device=x.device).unsqueeze(0) < lens.unsqueeze(1))  # (B,T)
    mse = ((x_hat - x)[mask]).pow(2).mean()
    kl  = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
    return mse + beta * kl


def plot_pca_embeddings(Z):
    pca = PCA(n_components=2)
    P = pca.fit_transform(Z.numpy())
    plt.figure()
    plt.scatter(P[:, 0], P[:, 1], s=5, alpha=0.6)
    plt.title("System embeddings (PCA)")
    plt.tight_layout()
    plt.show()

def plot_loss_curve(losses):
    plt.figure()
    plt.plot(losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss curve")
    plt.tight_layout()
    plt.show()
    

@torch.no_grad()
def autoencode_seq(m, seq):
    m.eval()
    x = seq.unsqueeze(0).to(device)             # (1,T,3)
    lens = torch.tensor([seq.shape[0]], device=device)
    x_hat, _, _ = m(x, lens)
    return x_hat.squeeze(0).cpu()[:seq.shape[0]]  # (T,3)


def plot_autoencoding_result(dataset, model, num_systems=10):

    first10 = [dataset.groups[i] for i in range(min(num_systems, len(dataset)))]
    recons = [autoencode_seq(model, s) for s in first10]

    plt.figure(figsize=(12,5))
    idx_map = {"a": 0, "r": 2, "mass": 1}
    for c,var in enumerate(["a","r","mass"]):
        plt.subplot(131+c)
        j = idx_map[var]
        for i, (orig, rec) in enumerate(zip(first10, recons)):
            L = orig.shape[0]
            xs = np.arange(L)
            if i == 0:
                pp = plt.plot(xs, orig[:, j], lw=1.2, alpha=0.8, label="orig")
                plt.plot(xs, rec[:, j],  lw=1.2, alpha=0.8, ls="--", label="recon", color=pp[0].get_color())
            else:
                pp = plt.plot(xs, orig[:, j], lw=1.0, alpha=0.5)
                plt.plot(xs, rec[:, j],  lw=1.0, alpha=0.5, ls="--", color=pp[0].get_color())
        plt.title(f"First systems • {var}")
        plt.xlabel("planet index (by distance)")
        plt.ylabel(var)
        plt.legend()
    plt.tight_layout()
    plt.show()
    
# %% 
NUM_ITERATIONS = 2 # TODO

losses = []
for _ in tqdm(range(NUM_ITERATIONS), desc="train"):
    for x, l in dataloader:
        x, l = x.to(device), l.to(device)
        opt.zero_grad()
        x_hat, mu, lv = model(x, l)
        loss = loss_batch(x, x_hat, l, mu, lv, beta=0.15)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
    # only one time per epoch
    losses.append(loss.item())

    print(f"epoch {_}: loss {loss.item():.4f}")

    if _ % 10 != 0:
        continue

    # --- embeddings (posterior mean) ---


    @torch.no_grad()
    def encode_all(m, loader):
        m.eval(); E=[]
        for x, l in loader:
            mu, lv = m.encode(x.to(device), l.to(device))
            E.append(mu.cpu())
        return torch.cat(E, 0)

    plot_loss_curve(losses)

    Z = encode_all(model, dataloader)  # (N_systems, latent_dim)

    plot_pca_embeddings(Z)
    plot_autoencoding_result(dataset, model, num_systems=10)

# %% plot autoencoding results and PCA of embeddings

Z = encode_all(model, dataloader)  # (N_systems, latent_dim)

plot_pca_embeddings(Z)

plot_autoencoding_result(dataset, model, num_systems=10)


# %% sample new systems with various latent samplers

# precompute empirical length dist once
lens_all = torch.tensor([len(g) for g in dataset.groups])
_vals, _cnt = torch.unique(lens_all, return_counts=True)
_probs = (_cnt / _cnt.sum()).numpy()

@torch.no_grad()
def fit_latent_mvn(model, loader, eps=1e-4):
    Z = encode_all(model, loader).cpu()                    # (N,D)
    m = Z.mean(0)
    C = torch.from_numpy(np.cov(Z.numpy(), rowvar=False)).float()
    C += eps * torch.eye(C.size(0))
    return torch.distributions.MultivariateNormal(m, covariance_matrix=C)

@torch.no_grad()
def fit_latent_gmm(model, loader, k=8, random_state=None):
    Z = encode_all(model, loader).cpu().numpy()
    return GaussianMixture(k, covariance_type="full", random_state=random_state).fit(Z)

def make_latent_sampler(model, loader, method="normal", k=8):
    D = model.mu.out_features
    if method == "normal":
        return lambda n: torch.randn(n, D, device=device)
    if method == "mvn":
        mvn = fit_latent_mvn(model, loader)
        return lambda n: mvn.sample((n,)).to(device)
    if method == "gmm":
        gmm = fit_latent_gmm(model, loader, k=k)
        return lambda n: torch.from_numpy(gmm.sample(n)[0]).float().to(device)
    raise ValueError(f"unknown sampler: {method}")

@torch.no_grad()
def sample_one(m, latent_sampler):
    L = int(np.random.choice(_vals.numpy(), p=_probs))
    z = latent_sampler(1)                                  # (1, D)
    return m.decode(z, L).squeeze(0).cpu()                 # (L,3)

# --- choose sampler: "normal" | "mvn" | "gmm" ---
latent_sampler = make_latent_sampler(model, dataloader, method="gmm", k=50)
#latent_sampler = make_latent_sampler(model, dataloader, method="mvn", k=5)


num_systems = 10

first = [dataset.groups[i] for i in range(min(num_systems, len(dataset)))]
recons = [autoencode_seq(model, s) for s in first]

idx_map = {"a": 0, "r": 2, "mass": 1}

plt.figure(figsize=(12,5))
for c,var in enumerate(["a","r","mass"]):
    plt.subplot(131+c)
    j = idx_map[var]
    for i, (orig, rec) in enumerate(zip(first, recons)):
        L = orig.shape[0]
        xs = np.arange(L)
        plt.plot(xs, orig[:, j], lw=1, alpha=0.5, label="orig")
    plt.title(f"First systems • {var}")
    plt.xlabel("planet index (by distance)")
    plt.ylabel(var)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,5))
for c, var in enumerate(["a","r","mass"]):
    j = idx_map[var]
    ax = plt.subplot(131+c)
    for _ in range(num_systems):
        sys = sample_one(model, latent_sampler)
        ax.plot(np.arange(sys.shape[0]), sys[:, j], lw=1.0, alpha=0.5)
    ax.set_title(f"Synthesized system • {var}")
    ax.set_xlabel("planet index (by distance)")
    ax.set_ylabel(var)
plt.tight_layout(); plt.show()

# %%
