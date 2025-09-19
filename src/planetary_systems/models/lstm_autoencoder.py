import torch
import torch.nn as nn

class SystemAutoencoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 embed_dim, 
                 num_layers, 
                 dropout):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=dropout)
        self.mu = nn.Linear(hidden_dim, embed_dim)
        self.logvar = nn.Linear(hidden_dim, embed_dim)
        self.from_embed = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers,
                               batch_first=True, dropout=dropout)

    def encode(self, x):
        _, (h, _) = self.encoder(x)
        h = h[-1]
        mu, logvar = self.mu(h), self.logvar(h)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)
        return z, mu, logvar

    def decode(self, z, seq_len):
        dec_in = self.from_embed(z).unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder(dec_in)
        return dec_out

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        dec_out = self.decode(z, x.size(1))
        return dec_out, z, mu, logvar


class HybridLoss(nn.Module):
    def __init__(self, alpha=10.0, beta=0.1):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.MSELoss(reduction="none")
        self.alpha = alpha
        self.beta = beta

    def forward(self, recon, target, mu, logvar):
        x_reg, x_cls = target[..., :-2], target[..., -2:]
        y_reg, y_cls = recon[..., :-2], recon[..., -2:]
        cls_tgt = x_cls.argmax(-1)
        cls_loss = self.cls_loss(y_cls.reshape(-1, 2), cls_tgt.reshape(-1))
        mask = x_cls[..., 0] > 0.5
        reg_loss = self.reg_loss(y_reg, x_reg)[mask].mean()
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return self.alpha * cls_loss + reg_loss + self.beta * kl