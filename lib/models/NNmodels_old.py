# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 11:21:02 2025

@author: Baaz
"""



# ---- Global and Refiner ----
class GlobalLatent(nn.Module):
    def __init__(self, dim=dim_parameter_encoder,initial_var=1):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(dim))
        initial_logvar = torch.log(torch.full((dim,), initial_var))
        self.logvar = nn.Parameter(initial_logvar)

        self.register_buffer('iteration', torch.tensor(1.0))

    def sample(self):
        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn_like(std)
        return self.mu + eps * std

    def update(self, z_all, gamma=None):
        if gamma is None:
            alpha = 0.6
            gamma = 1.0 / (self.iteration ** alpha)
        with torch.no_grad():
            batch_mu = z_all.mean(dim=0)
            batch_var = z_all.var(dim=0, unbiased=False).clamp(min=1e-6)
            batch_logvar = torch.log(batch_var)

            self.mu.data = (1 - gamma) * self.mu.data + gamma * batch_mu
            self.logvar.data = (1 - gamma) * self.logvar.data + gamma * batch_logvar
            self.iteration += 1.0

import torch.nn.functional as F

class IndividualRefiner(nn.Module):
    def __init__(self, input_dim=2 +  2*dim_parameter_encoder + 2 , output_dim=dim_parameter_encoder, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # Self-attention layers
        self.attn_query = nn.Linear(hidden_dim, hidden_dim)
        self.attn_key = nn.Linear(hidden_dim, hidden_dim)
        self.attn_value = nn.Linear(hidden_dim, hidden_dim)

        # Final projections for mean and log-variance of q(z|x)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_logvar = nn.Linear(hidden_dim, output_dim)
        
        for name, param in self.named_parameters():
           self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())
        
        self.register_buffer("iteration", torch.tensor(1.0))
        
    def update_ema(self, alpha=0.1):
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                ema_param.mul_(1 - alpha).add_(alpha * param.data)
            self.iteration += 1.0

    def apply_ema_weights(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                param.data.copy_(ema_param)


    def forward(self, t, x, log_sigma_add, log_sigma_prop, global_mu, global_logvar):
        """
        t: (batch_size, seq_len)
        x: (batch_size, seq_len)
        global_mu: (latent_dim,)
        global_logvar: (latent_dim,)
        """
    
        batch_size, seq_len = t.shape
        log_sigmas = torch.cat([log_sigma_add, log_sigma_prop])  # (2,)
    
        # Expand scalars to match batch and seq_len
        log_sigmas_expanded = log_sigmas.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)  # (batch_size, seq_len, 2)
        global_mu_expanded = global_mu.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)    # (batch_size, seq_len, latent_dim)
        global_logvar_expanded = global_logvar.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)  # same
    
        inp = torch.cat([
            t.unsqueeze(-1),  # (batch_size, seq_len, 1)
            x.unsqueeze(-1),  # (batch_size, seq_len, 1)
            log_sigmas_expanded,
            global_mu_expanded,
            global_logvar_expanded
        ], dim=-1)  # (batch_size, seq_len, input_dim)
    
        lstm_out, _ = self.lstm(inp)  # (batch_size, seq_len, hidden_dim)
    
        # Attention (batch version)
        Q = self.attn_query(lstm_out)
        K = self.attn_key(lstm_out)
        V = self.attn_value(lstm_out)
    
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)  # (batch_size, seq_len, seq_len)
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights, V)  # (batch_size, seq_len, hidden_dim)
    
        pooled = context.mean(dim=1)  # (batch_size, hidden_dim)
    
        mu_q = self.fc_mu(pooled)      # (batch_size, latent_dim)
        logvar_q = self.fc_logvar(pooled)  # (batch_size, latent_dim)
    
        return mu_q, logvar_q
