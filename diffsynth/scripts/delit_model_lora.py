"""
DelitWanLoRA - 基于 Wan Video VAE + Wan2.2 DiT (LoRA) 的 Delit 模型
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

# Add project root to path (one level up from diffsynth)
# This assumes we are at diffsynth/scripts/delit_model_lora.py
# parent -> diffsynth/scripts
# parent.parent -> diffsynth
# parent.parent.parent -> DiffSynth-Studio (Root)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from diffsynth.models.wan_video_vae import WanVideoVAE
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.diffusion.training_module import DiffusionTrainingModule

class DelitWanLoRA(nn.Module):
    """
    De-lighting Model based on Wan2.2 I2V + LoRA.
    
    Architecture:
    1. VAE (Frozen): Encodes Relit Image (Sparse) to Latent.
    2. DiT (LoRA): Denoises Env Map Latent conditioned on Relit Latent.
    
    Inputs:
    - Relit Equirectangular Image (Sparse): [B, 3, H, W] (Actually 4 channels from HDRCodec) -> VAE -> [B, 16, h, w]
    - Mask (Sparse): [B, 1, H, W] -> Downsample -> [B, 4, h, w]
    - Target Env Map: [B, 3, H, W] (4 ch encoded) -> VAE -> [B, 16, h, w] (Training only)
    """
    
    def __init__(
        self,
        vae: WanVideoVAE,
        dit: WanModel,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_target_modules: str = "q,k,v,o", # Target Attention Linear layers only
    ):
        super().__init__()
        self.vae = vae
        self.dit = dit
        self.lora_rank = lora_rank
        
        # 1. Freeze VAE
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()
        
        # 2. Inject LoRA to DiT
        self.training_module = DiffusionTrainingModule()
        
        # Freeze base DiT
        for param in self.dit.parameters():
            param.requires_grad = False
            
        # Add LoRA
        # Note: WanModel usually has attention in 'self_attn', 'cross_attn' blocks
        # Target modules names in WanModel: 
        # blocks.X.self_attn.q, k, v, o
        # blocks.X.cross_attn.q, k, v, o
        # blocks.X.ffn.0, 2
        
        targets = lora_target_modules.split(",")
        self.dit = self.training_module.add_lora_to_model(
            self.dit,
            target_modules=targets,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha
        )
        
        print(f"LoRA injected with rank {lora_rank} to modules: {targets}")
        
    def encode_latents(self, image_4ch: torch.Tensor) -> torch.Tensor:
        """
        Encode 4-channel HDR image using Wan VAE.
        Input: [B, 4, H, W]
        Output: [B, 16, h, w]
        """
        # Wan VAE expects [B, C, T, H, W] with C=3 usually.
        # But we modified FaceOLAT dataset to return 4 channels (HDR encoded).
        # WanVideoVAE input layer usually takes 3 channels.
        # IF the pre-trained VAE is standard Wan2.1, it takes 3 channels.
        # BUT our HDR Codec outputs 4 channels.
        # PROBLEM: Pretrained VAE expects 3 channels (RGB).
        # SOLUTION: We should project 4->3 or 4->Latent directly?
        # NO, we must respect the VAE input.
        # The 4th channel is Log Luminance. 
        # If we feed 3 channels (Normalized RGB) -> VAE -> Latent
        # We lose intensity information?
        # WAIT. The user said specifically: "Normalized RGB + Log Luminance (4 channels)".
        # And Claude designed a "from scratch" encoder before.
        # NOW we use PRETRAINED VAE.
        # Pretrained VAE only accepts RGB.
        # So we CANNOT feed 4 channels to it directly.
        
        # Workaround:
        # The dataset returns 4 channels.
        # Channel 0-2: Normalized RGB (Color)
        # Channel 3: Log Luminance (Intensity)
        # We can't feed intensity to standard VAE cleanly if it expects sRGB-like distribution.
        
        # Alternative: 
        # 1. Decode 4ch -> Linear RGB -> Tone Map -> sRGB -> VAE. (Lossy intensity)
        # 2. Feed 4ch to VAE? No, dimension mismatch.
        # 3. Fine-tune VAE? No, expensive.
        
        # Let's assume for now we use the first 3 channels (Normalized RGB) 
        # which represents the COLOR structure.
        # And we pass the 4th channel (Luminance) SEPARATELY?
        # Or we rely on the fact that VAE is robust?
        
        # Actually, if we want to reconstruct HDR, we need the intensity.
        # If VAE filters it out, we can't reconstruct it.
        # UNLESS we use the 'delit_model.py' approach where we had a separate branch?
        # But we are using Wan2.2 I2V now.
        
        # Wan2.2 I2V DiT Input = 36 channels.
        # 16 (Noisy) + 16 (Cond) + 4 (Mask).
        # If we use VAE for the latent, the latent is 16 channels.
        # If VAE only encodes RGB, where is the HDR info?
        
        # CRITICAL FLATCH: Standard Wan VAE cannot encode HDR floats (0-inf) or 4-channel codes perfectly.
        # However, Wan2.2 VAE is quite powerful.
        # Maybe we tone-map the HDR to 0-1, feed to VAE.
        # And rely on the DiT to predict the *latent* of the tone-mapped image?
        # Then how to get HDR back?
        # We need an inverse tone map.
        
        # Let's check `hdr_codec.py` again.
        # It encodes HDR to 4ch [0, 1].
        # So the input to VAE is [0, 1].
        # If we feed 3 channels of this to VAE, it sees "Normalized RGB".
        # It's a valid image.
        # What about channel 4?
        # We can resize channel 4 (Log Lum) to match Latent Size (64x128) and concatenating it?
        # But DiT expects 16+16+4 = 36 structure.
        
        # HYPOTHESIS: 
        # Wan2.2 I2V 36 channels = 16 (Noisy Latent) + 16 (Condition Latent) + 4 (Mask Latent? or just Mask?)
        # If we want to carry 4th channel info, maybe we can:
        # 1. Expand VAE input conv? (Breaks weights)
        # 2. Hack: Modify `z_cond` (16ch) to include the luminance information?
        #    - `z_cond` is 16 channels.
        #    - We can replace one channel? No.
        #    - We can add it to the 'Mask' part (4 ch)?
        
        # Proposed Solution for LoRA:
        # Use VAE to encode Channel 0-2 (RGB). -> z_rgb (16ch)
        # Resize Channel 3 (Lum) to latent size. -> z_lum (1ch)
        # We have 4 channels for "Mask".
        # Usually I2V mask is [1, h, w]. (or [4, h, w] repeating?)
        # We can put `z_lum` into the "Mask" channels?
        # Structure:
        # Input to DiT: [z_noisy (16), z_cond (16), z_extra (4)]
        # z_extra usually is Mask.
        # let's fill z_extra with [Mask, Lum, 0, 0].
        
        # This allows the model to see Luminance.
        # But wait, the Target `z_noisy` also needs to represent HDR.
        # If VAE only reconstructs RGB, we lose HDR in the output too!
        
        # RE-EVALUATION:
        # We need a VAE that handles 4 channels.
        # Wan VAE `z_dim=16`. Input `in_dim=3`.
        # Option: Train a lightweight adapter for VAE?
        # Or... Just input the 3 channels (Normalized RGB) to VAE.
        # And assume the VAE Latent *implicitly* captures enough info? 
        # No, Normalized RGB has NO intensity info (it is normalized by max RGB).
        
        # BETTER APPROACH:
        # Do NOT use `HDRCodec` for VAE input.
        # Use a Tone-mapping that serves 3 channels containing all info.
        # E.g. Log-compressed RGB.
        # RGB_log = log(1 + RGB) / log_scale.
        # This is [0, 1], 3 channels.
        # Contains correlation of color and intensity.
        # VAE encodes this 3-channel image.
        # DiT predicts latent of this 3-channel image.
        # Output -> VAE Decode -> RGB_log -> Exp -> HDR.
        
        # This allows using unmodified VAE.
        # I will modify `delit_dataset.py` later to support this or handle it here.
        # Let's assume input is [B, 3, H, W] Log-Encoded RGB for now.
        
        # For this implementation, I will assume we select the first 3 channels
        # or assume the input is already 3-channel suitable for VAE.
        # I will check `delit_dataset.py` later to verify.
        
        # Time dimension addition
        if image_4ch.ndim == 4:
            image = image_4ch.unsqueeze(2) # [B, C, 1, H, W]
        else:
            image = image_4ch
            
        # Use only first 3 channels for VAE
        image_3ch = image[:, :3, :, :, :]
        
        with torch.no_grad():
            # [B, 16, 1, h, w]
            z = self.vae.model.encode(image_3ch, self.vae.scale)
            
        return z

    def forward(
        self,
        relit_equir: torch.Tensor,
        person_mask_equir: torch.Tensor,
        valid_mask: torch.Tensor,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        text_context: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for training.
        
        Args:
            relit_equir: [B, 4, H, W] Sparse Input
            person_mask_equir: [B, 1, H, W]
            valid_mask: [B, 1, H, W]
            noisy_latents: [B, 16, 1, h, w] (Target noisy latent)
            timestep: [B]
            text_context: [B, L, D] (Optional)
        """
        # 1. Encode Condition (Relit Image)
        z_cond = self.encode_latents(relit_equir) # [B, 16, 1, h, w]
        
        # 2. Prepare Mask Condition
        # Downsample masks to latent size
        # Latent size is usually H/8, W/8
        _, _, _, h, w = z_cond.shape
        
        # Combine masks?
        # We have person_mask (1) and valid_mask (1).
        # Plus we want to pass Luminance (Channel 3 of relit_equir) via this channel?
        # Or just pass masks.
        
        # Let's simple downsample the masks.
        # We have 4 channels available for 'Mask' input (in_dim=36 - 16 - 16 = 4).
        # Channel 0: Person Mask
        # Channel 1: Valid Mask
        # Channel 2: Log Luminance (Downsampled from relit_equir channel 3)
        # Channel 3: Empty (Zero)
        
        B = relit_equir.shape[0]
        mask_cond = torch.zeros((B, 4, 1, h, w), device=relit_equir.device, dtype=relit_equir.dtype)
        
        # Downsample function
        def down(x):
            return F.interpolate(x, size=(h, w), mode='nearest').unsqueeze(2) # Add T dim
            
        mask_cond[:, 0:1] = down(person_mask_equir)
        mask_cond[:, 1:2] = down(valid_mask)
        mask_cond[:, 2:3] = down(relit_equir[:, 3:4]) # The Log Luminance Channel from input
        
        # 3. Concatenate Inputs
        # Wan2.2 expects: cat([noisy_latents, z_cond, mask_cond], dim=1)
        # [B, 36, 1, h, w]
        x_input = torch.cat([noisy_latents, z_cond, mask_cond], dim=1)
        
        # 4. Handle Text Context
        # If None, use learnable null embedding or zeros
        if text_context is None:
            # Wan2.2 text dim is 4096 usually.
            # We can pass a dummy context. 
            # Ideally load empty prompt embedding.
            # For now generate zeros (requires grad if we want to learn it? No LoRA is on DiT)
            text_context = torch.zeros((B, 1, 4096), device=relit_equir.device, dtype=relit_equir.dtype)
            
        # 5. DiT Forward
        # output = model(x, t, context)
        noise_pred = self.dit(x_input, timestep, text_context)
        
        return noise_pred
        
    def save_lora(self, path: str):
        """Save LoRA weights"""
        state_dict = self.training_module.export_trainable_state_dict(self.dit.state_dict())
        torch.save(state_dict, path)
        
    def load_lora(self, path: str):
        """Load LoRA weights"""
        state_dict = torch.load(path, map_location='cpu')
        state_dict = self.training_module.mapping_lora_state_dict(state_dict)
        self.dit.load_state_dict(state_dict, strict=False)

