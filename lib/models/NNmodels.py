
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class ODEFunc(nn.Module):
    """
    ODE function with optional EMA for parameters, dose handling, and skip connection.
    Supports multiple event types via `evid` (bolus=1, infusion=2, observation=0).
    """
    def __init__(self, dim_latent, dim_parameters_IC, dim_parameters_dynamic, hid_dim, drug_dim):
        super().__init__()
        self.drug_dim=drug_dim
        self.dim_latent = dim_latent
        self.dim_parameters_IC=dim_parameters_IC
        self.dim_parameters_dynamic = dim_parameters_dynamic
        self.fc1 = nn.Linear(1, hid_dim)
        self.SELU = nn.SELU()
        
        # Two separate layers for mean and log variance
        self.fc_mu = nn.Linear(hid_dim, dim_latent)
        self.fc_logvar = nn.Linear(hid_dim, dim_latent)
        
        
        
        # Deep network
        self.net = nn.Sequential(
            nn.Linear(dim_latent + dim_parameters_dynamic+drug_dim, hid_dim),
             nn.SELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SELU(),
           nn.Linear(hid_dim, dim_latent)
        )
        
        self.drug = nn.Sequential(
            nn.Linear(drug_dim, hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, drug_dim),
      
        )
        self.net_skip = nn.Sequential(
            nn.Linear(dim_latent + dim_parameters_dynamic+drug_dim, dim_latent)
        )

     
        # Noise parameter for dose pulses
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(0.05)))

        # EMA buffers
        for name, param in self.named_parameters():
            self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())
        self.register_buffer("iteration", torch.tensor(1.0))

    # === EMA methods ===
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

    def get_sigma(self):
        return torch.exp(self.log_sigma)
    

    def make_drug_masks(self, evid, n_drugs):
        """
        Create one mask per drug based on EVID.
        """
        n_drugs = int(n_drugs)
        masks = []
        for drug_id in range(1, n_drugs + 1):
            mask = (evid.long() == drug_id).float()
            masks.append(mask)
        return masks

    def bolus_pulse(self, t, dose_times, dose_amounts, dose_mask, evid, n_drugs):
        """
        Compute dose signal per drug using Gaussian pulses.
        Zero contribution if no doses exist for that individual/drug.
        Output: [batch, n_drugs]
        """
        sigma = self.get_sigma()
        batch_size = evid.size(0)
        device = t.device
        dose_signal = torch.zeros(batch_size, n_drugs, device=device)
    
        # Ensure t has shape [batch, num_times]
        if t.dim() == 1:
            t = t.unsqueeze(0).repeat(batch_size, 1)  # [batch, num_times]
    
        for drug_id in range(1, n_drugs + 1):
            # Mask doses for this drug
            mask = (evid == drug_id)  # [batch, num_doses]
            masked_times = dose_times * mask.float()
            masked_amounts = dose_amounts * mask.float()
    
            # Identify which batch elements actually have doses
            has_dose = mask.any(dim=1)  # [batch]
    
            if not has_dose.any():
                continue
    
            # Compute Gaussian pulse
            diff = t.unsqueeze(-1) - masked_times.unsqueeze(1)  # [batch, num_times, num_doses]
            gauss = torch.exp(-(diff / sigma)**2) * masked_amounts.unsqueeze(1)
            mask_nonnegative = diff >= 0
            pulse = gauss * mask_nonnegative.float()
    
            # Zero out contributions for batches with no doses
            pulse = pulse * has_dose.unsqueeze(-1).unsqueeze(-1).float()
    
            # Sum over doses
            dose_signal[:, drug_id - 1] = pulse.sum(dim=-1).squeeze()
         
        return dose_signal


        
    # def bolus_pulse(self, t, dose_times, dose_amounts, dose_mask, evid, n_drugs):
    #     """
    #     Compute dose signal per drug using interleaved dose arrays and EVID codes.
    #     """
    #     sigma=self.get_sigma()
    #     batch_size = evid.size(0)
    #     device = t.device
        
    #     diff = t.unsqueeze(-1) - dose_times  # [batch, num_times, num_doses]
    #     mask_nonnegative = diff >= 0

    #     gauss = torch.exp(-(diff / sigma)**2) * dose_amounts
 
     
    #     dose_signal=gauss*mask_nonnegative

      
    #     return dose_signal.sum(1)       
                
                    
            
        


    # === Forward pass ===
    def forward(self, t, x, cov, dose_times, dose_amounts, evid, dose_mask):
        batch_size = x.size(0)
        device = x.device
        
     

    
        # Compute dose input (bolus + infusion)
        bolus_signal = self.bolus_pulse(t, dose_times, dose_amounts, dose_mask, evid,self.drug_dim)
        dose_input = bolus_signal
        
        # Concatenate latent state and dose
        if dose_input.dim() == 1:
         dose_input = dose_input.unsqueeze(1)
         
        if t.dim() == 0:
            t = t.expand(batch_size, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1).expand(batch_size, 1)
       
        inp1 =torch.cat([x, self.drug(dose_input)], dim=1)

        dxdt_deep = self.net(inp1)
        dxdt_skip = self.net_skip(inp1)
        dxdt = dxdt_deep + dxdt_skip
       
        # Concatenate zeros for parameter dimensions
        zeros_param = torch.zeros(batch_size, self.dim_parameters_dynamic, device=device)
        dxdt_concat = torch.cat([dxdt, zeros_param], dim=1)
        return dxdt_concat



    

class TrainableNoise(nn.Module):
    """
    Estimates additive + optional proportional noise with EMA support.
    """
    def __init__(self, init_add_std=0.1, init_prop_std=0.0):
        super().__init__()
        size=1
        # Additive noise
        self.log_sigma_add = nn.Parameter(torch.full((size,), torch.log(torch.tensor(init_add_std))))

        # Optional proportional noise
        if init_prop_std > 0:
            self.use_prop = True
            self.log_sigma_prop = nn.Parameter(torch.full((size,), torch.log(torch.tensor(init_prop_std))))
        else:
            self.use_prop = False
            self.log_sigma_prop = None

        # EMA buffers
        self.register_buffer("log_sigma_add_ema", self.log_sigma_add.data.clone())
        if self.use_prop:
            self.register_buffer("log_sigma_prop_ema", self.log_sigma_prop.data.clone())
        self.register_buffer("iteration", torch.tensor(1.0))

    @property
    def sigma_add(self):
        return torch.exp(self.log_sigma_add)

    @property
    def sigma_prop(self):
        if not self.use_prop:
            return torch.zeros_like(self.log_sigma_add)
        return torch.exp(self.log_sigma_prop)

    def _compute_sigma_total(self, x_pred):
        sigma_add = self.sigma_add
        if self.use_prop:
            prop_term = (self.sigma_prop * x_pred.abs())  # shape == x_pred via broadcasting
            sigma_total = torch.sqrt(sigma_add**2 + prop_term**2)
        else:
            sigma_total = sigma_add
        return torch.clamp(sigma_total, min=1e-6)

    def forward(self, x_pred):
        sigma_total = self._compute_sigma_total(x_pred)
        eps = torch.randn_like(x_pred)
        return x_pred + sigma_total * eps

    def nll(self, x_true, x_pred, mask_L1=None, mask_missing_data=None):
        sigma_total = self._compute_sigma_total(x_pred)
        x_pred = torch.clamp(x_pred, min=-1e6, max=1e6)
        
        
       
        
        nll_elementwise = 0.5 * ((x_true - x_pred) / sigma_total) ** 2 + torch.log(sigma_total)
        mse_elementwise = (x_true - x_pred) ** 2
        
              
        
        #L1 for masked/dropped points
        if mask_L1 is not None:
            dropout_mask = mask_L1.bool()
            dropout_mask = dropout_mask.expand_as(nll_elementwise)

            if dropout_mask.any():
                L1_nll_elementwise = (x_true - x_pred).abs() #(2 ** 0.5) * (x_true - x_pred).abs() / sigma_total + torch.log(2 ** 0.5 * sigma_total)
              #  L1_nll_elementwise =  (x_true - x_pred).abs()

                L1_elementwise = (x_true - x_pred).abs()
         
                
                nll_elementwise[dropout_mask] = L1_nll_elementwise[dropout_mask]
                mse_elementwise[dropout_mask] = L1_elementwise[dropout_mask]

        # Apply mask
        if mask_missing_data is not None:
            nll_elementwise = nll_elementwise * mask_missing_data
            mse_elementwise = mse_elementwise * mask_missing_data

        # Per-individual sum
        nll_per_ind = nll_elementwise.sum(dim=1)
        mse_per_ind = mse_elementwise.sum(dim=1)
        
   #     print(nll_per_ind)
        return nll_per_ind.mean(), mse_per_ind.mean()

    def sample(self, x_pred, n_samples=1):
        sigma_total = self._compute_sigma_total(x_pred)
        if n_samples > 1:
            x_pred = x_pred.unsqueeze(0).expand(n_samples, *x_pred.shape)
            sigma_total = sigma_total.unsqueeze(0).expand(n_samples, *sigma_total.shape)
        eps = torch.randn_like(x_pred)
        return x_pred + sigma_total * eps

    # === EMA methods ===
    def update_ema(self, alpha=0.1):
        with torch.no_grad():
            self.log_sigma_add_ema.mul_(1 - alpha).add_(alpha * self.log_sigma_add)
            if self.use_prop:
                self.log_sigma_prop_ema.mul_(1 - alpha).add_(alpha * self.log_sigma_prop)
            self.iteration += 1.0

    def apply_ema_weights(self):
        with torch.no_grad():
            self.log_sigma_add.data.copy_(self.log_sigma_add_ema)
            if self.use_prop:
                self.log_sigma_prop.data.copy_(self.log_sigma_prop_ema)









class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttentionPooling(nn.Module):
    """Multi-head attention pooling layer with learnable queries."""
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.randn(num_heads, d_model))
        self.scale = math.sqrt(d_model)

    def forward(self, h, mask):
        """
        h: [B, T, D]
        mask: [B, T] boolean (True = valid)
        Returns: [B, H*D]
        """
        B, T, D = h.shape
        H = self.num_heads
        scores = torch.einsum("btd,hd->bth", h, self.query) / self.scale
        scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        attn = torch.softmax(scores, dim=1)  # [B, T, H]
        pooled = torch.einsum("btd,bth->bhd", h, attn)
        return pooled.reshape(B, H * D)


class MHAEncoderLayer(nn.Module):
    """Transformer encoder layer (batch_first=True)."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation=nn.SELU()):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                         dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout2(ffn_out))


class Encoder_Transformer(nn.Module):
    """
    Transformer-based VAE encoder with full-covariance posterior and prior.
    Supports learnable prior mean and covariance.
    """
    def __init__(
        self,
        dim_latent,
        dim_parameters_IC,
        dim_parameters_dynamic,
        dim_cov,
        model_dim=64,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        cov_diag_epsilon=1e-5,
        learn_prior_mean=True,
        learn_prior_covariance=False,
        diagonal_only=False,
        IC_dose_dependent=False
    ):
        super().__init__()
        input_dim = 2  # (t, x)
        self.dim_latent = dim_latent
        self.dim_parameter_IC = dim_parameters_IC
        self.dim_parameter_dynamic = dim_parameters_dynamic
        self.IC_dose_dependent=IC_dose_dependent
        self.total_parameters = dim_parameters_IC+dim_parameters_dynamic
        self.dim_cov=dim_cov
        
        self.cov_diag_epsilon = cov_diag_epsilon
        self.diagonal_only = diagonal_only
        self.selu = nn.SELU()
        self.learn_prior_mean = learn_prior_mean
        self.learn_prior_covariance = learn_prior_covariance

        # Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, model_dim),
        )

        # Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads,
            dim_feedforward=model_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj1 = nn.Linear(input_dim, hidden_dim)
        self.input_proj2 = nn.Linear(hidden_dim, model_dim)

        # Pooling
        self.pooler = MultiHeadAttentionPooling(model_dim, num_heads=num_heads)
        self.pool_norm = nn.LayerNorm(model_dim * num_heads)

        # Posterior mean
        self.fc_mu1 = nn.Linear(model_dim * num_heads, hidden_dim)
        self.fc_mu2 = nn.Linear(hidden_dim, self.total_parameters)
        
     

        

        # Posterior covariance
        self.fc_A1 = nn.Linear(model_dim * num_heads, hidden_dim)
        self.fc_A2 = nn.Linear(hidden_dim, self.total_parameters * self.total_parameters)
        with torch.no_grad():
            self.fc_A2.bias.copy_(torch.eye(self.total_parameters).flatten())
        

        self.z0_transformed=nn.Sequential(
             nn.Linear(dim_latent+dim_parameters_IC, 32),
             nn.SELU(),
       nn.Linear(32, dim_latent),
         )
        self.cov_transform = nn.Linear(self.dim_cov + model_dim * num_heads, model_dim * num_heads)
        self.cov_embeddings = nn.Linear(self.dim_cov, self.dim_cov)

        # NODE initial condition
        if IC_dose_dependent:
            self.z0_mu=nn.Sequential(
                 nn.Linear(1, 32),
                 nn.SELU(),
                 nn.Linear(32, 32),
                 nn.SELU(),
                 nn.Linear(32, 32),
           nn.Linear(32, dim_latent),
             )
            
            
            

            
        else: 

            self.z0_mu = nn.Parameter(torch.zeros(dim_latent))

         
        if learn_prior_mean:
            self.mu_p=nn.Sequential(
                 nn.Linear(1+self.dim_cov, 32),
                 nn.SELU(),
           nn.Linear(32, self.total_parameters),
             )
            
            
            
            
        else:
            self.register_buffer("mu_p", torch.zeros(self.total_parameters))

        if learn_prior_covariance:
            init_std = 0.01
            if diagonal_only:
                self.prior_A = nn.Parameter(torch.ones(self.total_parameters))

            else:    
                
                self.prior_A = nn.Parameter(
                        torch.eye(self.total_parameters) + torch.randn(self.total_parameters, self.total_parameters) * init_std
                    )
                self.cov_to_A = nn.Sequential(
                    nn.Linear(self.dim_cov, 32),
                    nn.ReLU(),
                    nn.Linear(32, self.total_parameters * self.total_parameters)
                )


        else:
            self.register_buffer("prior_A", torch.eye(self.total_parameters))
   
        
        # EMA buffers
        for name, param in self.named_parameters():
            self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())
        self.register_buffer("iteration", torch.tensor(1.0))

    # ---------------- EMA helpers ----------------
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

    # ---------------- Prior ----------------
    def get_prior(self,cov , dose_tensor,learn_prior_mean, batch_size=None):
       D = self.total_parameters
       device = self.prior_A.device

       if self.learn_prior_covariance:
           delta_A = self.cov_to_A(cov)  # [B, D*D]
           delta_A = delta_A.view(batch_size, D, D)  # reshape to [B, D, D]

           prior_base = self.prior_A.unsqueeze(0) + delta_A  # [B, D, D]

           Sigma_p = prior_base @ prior_base.mT + self.cov_diag_epsilon * torch.eye(D, device=device)



       else:
           Sigma_p = torch.eye(D, device=device)

       L_p = torch.linalg.cholesky(Sigma_p)
       x=torch.cat([cov,dose_tensor[:, 0:1] ], dim=1)
       if learn_prior_mean:
           mu_p = self.mu_p(x)
       else:
           mu_p=self.mu_p
           if batch_size is not None:
               mu_p = mu_p.unsqueeze(0).expand(batch_size, -1)

       
      # if batch_size is not None:
        #    L_p = L_p.unsqueeze(0).expand(batch_size, -1, -1)
          #  if mu_p.dim() == 1:
                

       return mu_p, L_p
    def _sample_posterior(self, mu_q, L_q, num_samples=1):
       B, D = mu_q.shape
       device = mu_q.device
   
       if num_samples > 1:
           eps = torch.randn(B, D, num_samples, device=device)  # [B, D, S]
           samples = mu_q.unsqueeze(1) + torch.einsum("bij,bjs->bis", L_q, eps).permute(0, 2, 1)  # [B, S, D]
           return samples.reshape(B * num_samples, D)  # flatten into [B*S, D]
       else:
           eps = torch.randn(B, D, device=device)  # [B, D]
           return mu_q + torch.einsum("bij,bj->bi", L_q, eps)  # [B, D]



   
    # ---------------- Forward ----------------
    def forward(self, t, x, cov, dose_tensor, mask=None, only_median=False,enable_ae=False, num_samples=1, min_batch_size=None, augment=False,sample_posterior=True):
   
            
   
        B, T = t.shape
        device = t.device
        if mask is None:
           mask = torch.ones(B, T, dtype=torch.bool, device=device)
        else:
           B = len(mask)
           T = max(len(m) for m in mask)  # or global_max_len
           mask_tensor = torch.zeros(B, T, dtype=torch.bool, device=mask[0].device)
           
           # 2. Fill in True where data exists
           for i, m in enumerate(mask):
               mask_tensor[i, :len(m)] = m  # m is 1D bool tensor

           mask=mask_tensor
       
      
        # ------------------ Posterior ------------------
        inp = torch.stack([t, x], dim=-1)
        
        h = self.input_proj(inp)
        h = self.pos_encoder(h)
      

        
        h_enc = self.transformer(h, src_key_padding_mask=~mask)
     
        # Masked pooling
        pooled = self.pooler(h_enc, mask)
        pooled = self.pool_norm(pooled)
        pooled = F.dropout(pooled, p=0.1, training=self.training) #+ torch.randn_like(pooled)
        cov_embeddings = self.cov_embeddings(cov)


        pooled = self.cov_transform(torch.cat([pooled, cov_embeddings], dim=1))


     
        # Posterior mean
        mu_q = self.fc_mu2 (self.fc_mu1(pooled))
       
        # Posterior covariance
      
        A_q_flat = self.fc_A2(self.selu(self.fc_A1(pooled)))  # [B, D*D]
    
        A_q = A_q_flat.view(B, self.total_parameters, self.total_parameters)                  # [B, D, D]
        Sigma_q = A_q @ A_q.transpose(-1, -2) + self.cov_diag_epsilon * torch.eye(self.total_parameters, device=A_q.device)

        L_q = torch.linalg.cholesky(Sigma_q)          # Cholesky factor


    
        # ------------------ Prior ------------------
        mu_p, L_p = self.get_prior(cov,dose_tensor,self.learn_prior_mean, batch_size=B)
        # ------------------ Median-only ------------------
        if only_median:
            k_params = mu_p #torch.zeros(B, self.total_parameters, device=device)
      
            mu_q=mu_p
            L_p = torch.eye(self.total_parameters, device=device).unsqueeze(0).expand(B, self.total_parameters, self.total_parameters)
            L_q=L_p
            
            mask = torch.ones(B, 1, device=device)

     
        else:
        # ------------------ Autoencoder ------------------
            if enable_ae: 
                k_params=mu_q
                mu_q=mu_q

                L_q=L_p
                mask = torch.zeros(B, 1, dtype=torch.bool, device=device)

  
            # ------------------ Variational Autoencoder ------------------    
            else:  

                if sample_posterior: # Sample from posterior (during training, individual predictions)
                    k_params=self._sample_posterior(mu_q, L_q, num_samples=num_samples)
                    mask = torch.zeros(B*num_samples, 1, dtype=torch.bool, device=device)                
                else:                # Sample from the prior (in VPC e.g.,)
                   
                 
                    k_params=self._sample_posterior(mu_p, L_p, num_samples=num_samples)
                    mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
                    
                   
         

        # ------------------ Split k into dynamic and IC -----------------
        dyn_start = self.dim_parameter_IC
        dyn_end = self.total_parameters

        
        k_params_IC =k_params[:, 0:dyn_start]  
        k_params_K = k_params[:, dyn_start:dyn_end] 

        # ------------------ Dose-dependent IC ------------------
        if self.IC_dose_dependent:
            dose_input = dose_tensor[:, 0:1]  # shape [B, 1]
            z0_base = self.z0_mu(dose_input)  # [B, dim_latent]
         
      
         # ------------------ Dose-independent IC ------------------    
        else: 
            z0_base = self.z0_mu  # [B, dim_latent]
            z0_base = z0_base.repeat(B, 1)   # shape: [B, dim_latent]
 
                
 
    
        # ------------------ Multiple Monte Carlo samples ------------------
        if num_samples > 1:
            z0 = z0_base.unsqueeze(1).expand(-1, num_samples, -1).reshape(B * num_samples, -1) 
        else:
            z0 = z0_base
            

        # ------------------ Variability on IC -----------------
        if self.dim_parameter_IC > 0:
            z0 = self.z0_transformed(torch.cat([z0, k_params_IC],dim=-1))
                  


        
        k_params=torch.cat([z0, k_params_K], dim=-1)
        return k_params, z0, mu_q, L_q, mu_p, L_p,  mask







class SimpleDecoder(nn.Module):
    def __init__(self, dim_latent):
        super().__init__()
        self.dim_latent=dim_latent
        
        # Define the network
        self.net = nn.Sequential(
            nn.Linear(dim_latent, 1))
        
   

        # EMA buffers
        for name, param in self.named_parameters():
            self.register_buffer(f"{name.replace('.', '_')}_ema", param.data.clone())
        self.register_buffer("iteration", torch.tensor(1.0))


    def forward(self, z):
        return self.net(z).squeeze(-1)

    def update_ema(self, alpha=0.1):
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                ema_param = ema_param.to(param.device)
                ema_param.mul_(1 - alpha).add_(alpha * param.data)
                setattr(self, f"{name.replace('.', '_')}_ema", ema_param)
            self.iteration += 1.0

    def apply_ema_weights(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema_param = getattr(self, f"{name.replace('.', '_')}_ema")
                param.data.copy_(ema_param)

