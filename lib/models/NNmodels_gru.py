# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 00:53:34 2025

@author: Baaz
"""

# ---- ODE ----
class ODEFunc(nn.Module):
    def __init__(self, latent_dim, dim_parameter_encoder, hid_dim):
        super().__init__()
        self.dim_parameter_encoder = dim_parameter_encoder  
        self.dim_latent = latent_dim
        self.net = nn.Sequential(
            nn.Linear((latent_dim+dim_parameter_encoder+2), hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, latent_dim ))
        
        self.skip = nn.Sequential(
            nn.Linear((latent_dim+dim_parameter_encoder), latent_dim))
        
        self.beta = nn.Sequential(
            nn.Linear(2, hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, latent_dim ))
        
        self.gamma = nn.Sequential(
            nn.Linear(2, hid_dim),
            nn.SELU(),
            nn.Linear(hid_dim, latent_dim ))


        self.attention = DoseAttention(hidden_dim=16)

        self.gru_output_proj = nn.Sequential(
                    nn.Linear(16, 16),
              nn.SELU(),
              nn.Linear(16, 16),
                      )
        self.gru = nn.GRU(input_size=2, hidden_size=16, batch_first=True)
 
        self.dose_proj = nn.Linear(16, 2)  # project to latent_dim
    

        self.linear1 = nn.Sequential(nn.Linear(1, 16), nn.SELU(), nn.Linear(16, 1))
        self.linear2 = nn.Sequential(nn.Linear(1, 16), nn.SELU(), nn.Linear(16, 1))
   

   
    def compute_T_all(self, t, dose_times, dose_mask):
        t_scalar = t.item()
        
        # Squeeze last dim if it's singleton
        if dose_times.dim() == 3 and dose_times.size(-1) == 1:
            dose_times = dose_times.squeeze(-1)  # shape becomes [10,4]
        
        delta_t = t_scalar - dose_times  # [10,4]
        T_all = torch.where(dose_mask & (dose_times <= t_scalar), delta_t, torch.zeros_like(delta_t))
        return T_all  # shape: [10,4]
    
    def T(self, t, dose_times):
        t_scalar = t.item()
        relevant_doses = dose_times[dose_times <= t_scalar]
        if len(relevant_doses) == 0:
            return t_scalar
        else:
            return t_scalar - relevant_doses.max().item()
        

  

    def forward(self, t, x, dose_times,dose_amounts, dose_mask):
        batch_size = x.size(0)
        device = x.device
    
        t_scalar = t.item()
     
        T_all = self.T(t, dose_times)  # time since each dose
        T = T_all * torch.ones(batch_size, 1, device=x.device)  # (batch_size,1)
       # T2_all=self.compute_T_all(t, dose_times,dose_mask) 

       # new_dose_mask = (T2_all > 0)

     
        dose_amounts_squeezed = dose_amounts.squeeze(-1)  # [10, 4]
     
       # gru_input = torch.cat([T2_all.unsqueeze(-1), dose_amounts_squeezed.unsqueeze(-1)], dim=-1)
# shape: [batch, num_doses, 2]

      #  mask_expanded = new_dose_mask.unsqueeze(-1).float()  # [2, 4, 1]
     #   print(self.gru_output_proj.bias)

      #  gru_output, _ = self.gru(gru_input)  # [B, T, H]
      #  masked_gru_output = gru_output 
       # print(gru_output)
        #print(gru_output)
     #   projected = self.gru_output_proj(masked_gru_output)  # [B, T, 2]
        
         # zero invalid timesteps
     

       # attn_output, attn_weights = self.attention(projected* mask_expanded , new_dose_mask,T2_all)
        
     #   agg = attn_output  # [B, 2], dose embedding weighted by attention

        
       # out1 = self.linear1(agg[:, 0].unsqueeze(-1))  # [batch, 1]
       # out2 = self.linear2(agg[:, 1].unsqueeze(-1))  # [batch, 1]

       # final = torch.cat([out1, out2], dim=1)  # [batch, 2]
     
     #   t_tensor = torch.full((batch_size, 1), t.item(), device=x.device)

      #  dose_single=dose_amounts_squeezed[:, 0]
        
       # T_tensor = torch.full((batch_size, 1), T_all, device=x.device)
       #print(dose_amounts)
        dose_exp = dose_amounts_squeezed[ :,0].unsqueeze(1)  # shape [4, 1]


 
  
        inp =  torch.cat([x,  T, dose_exp], dim=1)
       # inp2 = final
       # gamma=self.gamma(final)
       # beta=self.beta(final)
        
        dxdt_deep = self.net(inp) #+beta
        
        #dxdt_skip=self.skip(inp)
        
        dxdt=dxdt_deep#+dxdt_skip
        
        zero = torch.zeros(batch_size, self.dim_parameter_encoder, device=device)
        dxdt_concat = torch.cat([dxdt, zero], dim=1)  # shape: [batch_size, latent_dim]
       # print(t_scalar)
     #   print(new_dose_mask)
     
     #   print(dose_single.unsqueeze(1))
       # print(attn_output)
       # print(attn_weights)
       
       
        return dxdt_concat