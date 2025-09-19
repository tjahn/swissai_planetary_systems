import torch
import torch.nn as nn

class TransEncGRUVAE(nn.Module):
        def __init__(self, d_model=64, nhead=4, num_layers=2, dim_ff=128, latent=16, p=0.2, dec_layers=1):
            super().__init__()
            self.inp = nn.Linear(3, d_model)
            self.drop_in   = nn.Dropout(p)
            self.drop_head = nn.Dropout(p)
            self.drop_dec  = nn.Dropout(p)

            enc_layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_ff, dropout=p, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers)

            self.mu = nn.Linear(d_model, latent)
            self.lv = nn.Linear(d_model, latent)

            self.dec_in = nn.Linear(latent, d_model)
            self.dec = nn.GRU(d_model, d_model, dec_layers, batch_first=True,
                            dropout=(p if dec_layers > 1 else 0.0))
            self.out = nn.Linear(d_model, 3)

        def encode(self, x, lens):
            T = x.size(1)
            pad_mask = (torch.arange(T, device=x.device).unsqueeze(0) >= lens.unsqueeze(1))
            h = self.encoder(self.drop_in(self.inp(x)), src_key_padding_mask=pad_mask)  # (B,T,D)
            denom = lens.clamp(min=1).unsqueeze(1)
            h = h.masked_fill(pad_mask.unsqueeze(-1), 0).sum(1) / denom                 # (B,D)
            h = self.drop_head(h)
            return self.mu(h), self.lv(h)

        @staticmethod
        def reparam(mu, lv):
            return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)

        def decode(self, z, T):
            B = z.size(0)
            h0 = self.dec_in(z).unsqueeze(0)
            zeros = torch.zeros(B, T, h0.size(-1), device=z.device)
            y, _ = self.dec(zeros, h0)
            y = self.drop_dec(y)
            return self.out(y)

        def forward(self, x, lens):
            mu, lv = self.encode(x, lens)
            z = self.reparam(mu, lv)
            x_hat = self.decode(z, x.size(1))
            return x_hat, mu, lv
