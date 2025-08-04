from abc import ABC
from models.estimator import DiT
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.modules.audio_embedding import AudioEmbedding
from .scheduler import PolynomialConvexScheduler
from contextlib import nullcontext
from torch import Tensor
from typing import Tuple
from einops import rearrange

class BaseFDFD(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if hasattr(config, "sigma_min"):
            self.sigma_min = config.sigma_min
        else:
            self.sigma_min = 1e-4
 
    def forward(self, content_emb, prompt_codes, spk_emb, n_timesteps):
        audio_length = content_emb.shape[2]
        x_init = self.source_distribution.sample(tensor_size=(1, self.n_quantizers * audio_length),
                                                 device=content_emb.device)

        prompt_emb = self.get_prompt_emb(prompt_codes)  # [B, 6, L_p, D]

        codes = self.solver_euler(x=x_init,
                                  content_emb=content_emb,
                                  prompt_emb=prompt_emb,
                                  spk_emb=spk_emb,
                                  n_timesteps=n_timesteps)
        codes = rearrange(codes, 'b (n l) -> n b l', n=self.n_quantizers)
        return codes

    def solver_euler(self, x, content_emb, prompt_emb, spk_emb, n_timesteps):
        t_span = torch.linspace(0, 1 - self.sigma_min, n_timesteps + 1)
        x_t = x.clone()
        steps_counter = 0
        ctx = nullcontext()
 
        def categorical(probs):
            r"""Categorical sampler according to weights in the last dimension of ``probs`` using :func:`torch.multinomial`.
 
            Args:
                probs (Tensor): probabilities.
 
            Returns:
                Tensor: Samples.
            """
 
            return torch.multinomial(probs.flatten(0, -2), 1, replacement=True).view(
                *probs.shape[:-1]
            )
        
        with ctx:
            for i in range(n_timesteps):
                t = t_span[i : i + 1]
                h = t_span[i + 1 : i + 2] - t_span[i : i + 1]
 
                # Sample x_1 ~ p_1|t( \cdot |x_t)
                xt_emb = self.audio_embed(x_t).to(content_emb.device) # [B, 6L, D]
                xt_emb = rearrange(xt_emb, 'b (n l) d -> b n l d', n=self.n_quantizers) # [B, L, 6D]
                p_1t = self.estimator(x=xt_emb,
                                      content_emb=content_emb,
                                      prompt_emb=prompt_emb,
                                      spk_emb=spk_emb,
                                      t=t.repeat(x_t.shape[0]).to(content_emb.device))

                p_1t = self.get_logits(rearrange(p_1t, 'b l (n d) -> b n l d', n=self.n_quantizers))
                p_1t = rearrange(p_1t, 'b n l d -> b (n l) d', n=self.n_quantizers)
                p_1t = torch.softmax(p_1t, dim=-1)
                
                x_1 = categorical(p_1t.to(dtype=torch.float64))
 
                # Checks if final step
                if i == n_timesteps - 1:
                    x_t = x_1
                else:
                    # Compute u_t(x|x_t,x_1)
                    scheduler_output = self.scheduler(t=t)

                    k_t = scheduler_output.alpha_t.to(content_emb.device)
                    d_k_t = scheduler_output.d_alpha_t.to(content_emb.device)

                    delta_1 = F.one_hot(x_1, num_classes=self.vocab_size).to(
                        k_t.dtype
                    )
 
                    u = d_k_t / (1 - k_t) * delta_1
 
                    # Set u_t(x_t|x_t,x_1) = 0
                    delta_t = F.one_hot(x_t, num_classes=self.vocab_size)
                    u = torch.where(
                        delta_t.to(dtype=torch.bool), torch.zeros_like(u), u
                    )
 
                    # Sample x_t ~ u_t( \cdot |x_t,x_1)
                    intensity = u.sum(dim=-1)  # Assuming u_t(xt|xt,x1) := 0
                    mask_jump = torch.rand(
                        size=x_t.shape, device=x_t.device
                    ) < 1 - torch.exp(-h.to(x_t.device) * intensity)
 
                    if mask_jump.sum() > 0:
                        x_t[mask_jump] = categorical(
                            u[mask_jump].to(dtype=torch.float64)
                        )
 
                steps_counter += 1
                t = t + h
        return x_t
    
    def interpolate(self, x1, x0, t):
        sigma_t = self.scheduler(t).sigma_t
        dim_diff = x1.ndim - t.ndim
        t_expanded = t.clone()
        t_expanded = t_expanded.reshape(-1, *([1] * dim_diff))
        sigma_t = t_expanded.expand_as(x1)
        is_pad = (x1 == self.pad_token_id)
        source_indices = (torch.rand(size=x1.shape, device=x1.device) < sigma_t) & (~is_pad)
        xt = torch.where(condition=source_indices, input=x0, other=x1)
        return xt
    
    def get_logits(self, hidden_states):
        # hidden_states [B, 4, L, D]
        chunks = torch.split(hidden_states, [1, 3], dim=1)
        logits_list = [head(chunk) for head, chunk in zip(self.linear_heads, chunks)]
        logits = torch.cat(logits_list, dim=1) # [B, 6, L, vocab]
        return logits
    
    def get_prompt_emb(self, prompts):
        # prompts: [B, 6, L_p]
        prompt_prosody_codes, prompt_acoustic_codes = prompts[:, :1], prompts[:, 3:]
        prompts = torch.cat([prompt_prosody_codes, prompt_acoustic_codes], dim=1) # [B, 4, L_p]
        prompts = rearrange(prompts, 'b n l -> b (n l)', n=self.n_quantizers)
        prompt_emb = self.audio_embed(prompts) # [B, 4L_p, D]
        prompt_emb = rearrange(prompt_emb, 'b (n l) d -> b n l d', n=self.n_quantizers)
        B, _, L, D = prompt_emb.shape
        prompt_emb = torch.cat([
            prompt_emb[:, :1],
            torch.zeros((B, 2, L, D), device=prompt_emb.device, dtype=prompt_emb.dtype),
            prompt_emb[:, 1:]
        ], dim=1)

        return prompt_emb
 
    def compute_loss(
        self,
        x1, # [B, 4, L]
        prompts, # [B, 6, L_p]
        content_emb, # [B, 2, L, D]
        spk_embs, # [B, spk_dim]
        audio_masks # [B, L]
    ):
        batch_size = x1.shape[0]
        x1 = rearrange(x1, 'b n l -> b (n l)', n=self.n_quantizers) # [B, 4L]

        with torch.no_grad():
            x0 = self.source_distribution.sample_like(x1) # [B, 4L]

        t = torch.rand((batch_size,), dtype=content_emb.dtype, device=content_emb.device)

        xt = self.interpolate(x1, x0, t) # [B, 4L]
        xt_emb = self.audio_embed(xt) # [B, 4L, D]
        xt_emb = rearrange(xt_emb, 'b (n l) d -> b n l d', n=self.n_quantizers) # [B, 4, L, D]

        prompt_emb = self.get_prompt_emb(prompts) # [B, 6, L_p, D]
 
        output = self.estimator(x=xt_emb,
                                content_emb=content_emb,
                                prompt_emb=prompt_emb,
                                spk_emb=spk_embs,
                                t=t,
                                audio_masks=audio_masks) # [B, L, 6D]
        
        last_hidden_state = rearrange(output, 'b l (n d) -> b n l d', n=self.n_quantizers) # [B, 6, L, D]   
        logits = self.get_logits(last_hidden_state)  # [B, 6, L, Vocab]

        fdfd_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            x1.view(-1),
            ignore_index=self.pad_token_id
        )

        return {"FDFD_Loss": fdfd_loss}

class FDFD(BaseFDFD):
    """Factorized Discrete Flow Denoiser"""   
    def __init__(self, config):
        super().__init__(config)
        self.pad_token_id = config.vocab_size
        self.sep_token_id = config.vocab_size + 1
        self.mask_token_id = config.vocab_size + 2
        self.vocab_size = config.vocab_size + 3
        self.n_quantizers = config.n_quantizers

        self.scheduler = PolynomialConvexScheduler(n=config.exponent)
        self.source_distribution = MaskedSourceDistribution(mask_token=self.mask_token_id)
        
        config.estimator.n_quantizers = self.n_quantizers
        self.estimator = DiT(**config.estimator)
        self.audio_embed = AudioEmbedding(self.vocab_size, config.estimator.hidden_dim, self.pad_token_id)

        self.linear_heads = nn.ModuleList([
            nn.Linear(config.estimator.hidden_dim, self.vocab_size) for _ in range(2)
        ])
 
class SourceDistribution(ABC):
    def __init__(
        self,
    ) -> None:
        ...
 
    def sample(self, tensor_size: Tuple[int, ...], device: torch.device) -> Tensor:
        ...
 
    def sample_like(self, tensor_like: Tensor) -> Tensor:
        ...
 
class MaskedSourceDistribution(SourceDistribution):
    def __init__(self, mask_token: int) -> None:
        self.mask_token = mask_token
 
    @property
    def masked(self) -> bool:
        return True
 
    def sample(self, tensor_size: Tuple[int, ...], device: torch.device) -> Tensor:
        return torch.zeros(tensor_size, device=device).fill_(self.mask_token).long()
 
    def sample_like(self, tensor_like: Tensor) -> Tensor:
        return torch.zeros_like(tensor_like).fill_(self.mask_token).long()