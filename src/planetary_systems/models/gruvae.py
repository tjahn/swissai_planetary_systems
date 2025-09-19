import torch
import torch.nn as nn

class GRUVAE(nn.Module):
    def __init__(self, 
                 d_hidden=64,
                 latent=25,
                 dropout=0.1, 
                 enc_layers=2, 
                 dec_layers=2
                 ):
        super().__init__()
        self.inp = nn.Linear(3, d_hidden)
        self.drop_in   = nn.Dropout(dropout)
        self.drop_head = nn.Dropout(dropout)   # before mu/lv
        self.drop_dec  = nn.Dropout(dropout)   # on decoder outputs

        self.enc = nn.GRU(d_hidden, d_hidden, num_layers=enc_layers, batch_first=True,
                          dropout=(dropout if enc_layers > 1 else 0.0))
        self.mu  = nn.Linear(d_hidden, latent)
        self.lv  = nn.Linear(d_hidden, latent)

        self.dec_in = nn.Linear(latent, d_hidden)
        self.dec = nn.GRU(d_hidden, d_hidden, num_layers=dec_layers, batch_first=True,
                          dropout=(dropout if dec_layers > 1 else 0.0))
        self.out = nn.Linear(d_hidden, 3)

    def encode(self, x, lens):
        x = self.drop_in(self.inp(x))
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.enc(packed)                  # (enc_layers,B,H)
        h = h[-1]                                # last layer (B,H)
        h = self.drop_head(h)
        return self.mu(h), self.lv(h)

    @staticmethod
    def reparam(mu, lv):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)

    def decode(self, z, T):
        B = z.size(0)
        h0 = self.dec_in(z).unsqueeze(0)             # (1,B,H)
        h0 = h0.repeat(self.dec.num_layers, 1, 1)    # (dec_layers,B,H)
        zeros = torch.zeros(B, T, h0.size(-1), device=z.device)
        y, _ = self.dec(zeros, h0)
        y = self.drop_dec(y)
        return self.out(y)

    def forward(self, x, lens):
        mu, lv = self.encode(x, lens)
        z = self.reparam(mu, lv)
        x_hat = self.decode(z, x.size(1))
        return x_hat, mu, lv