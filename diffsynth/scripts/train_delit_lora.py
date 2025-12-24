import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
# __file__ -> diffsynth/scripts/train_delit_lora.py
# parent -> diffsynth/scripts
# parent.parent -> diffsynth
# parent.parent.parent -> DiffSynth-Studio (Root)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from diffsynth.scripts.delit_dataset import FaceOLATDelitDataset
from diffsynth.scripts.delit_model_lora import DelitWanLoRA
from diffsynth.models import model_loader
from diffsynth.diffusion.flow_match import FlowMatchScheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Train Delit Wan2.2 LoRA")
    parser.add_argument("--data_root", type=str, required=True, help="Path to FaceOLAT dataset")
    parser.add_argument("--output_dir", type=str, default="output/delit_lora", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (per GPU)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save interval (steps)")
    parser.add_argument("--accumulate_grad", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--wan_dit_path", type=str, required=True, help="Path to Wan2.2 DiT model file")
    parser.add_argument("--wan_vae_path", type=str, required=True, help="Path to Wan2.2 VAE model file")
    return parser.parse_args()

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Models
    print("Loading models...")
    # Use ModelPool to load specific files
    pool = model_loader.ModelPool()
    
    # Load VAE (Frozen)
    pool.auto_load_model(args.wan_vae_path)
    vae = pool.fetch_model("wan_video_vae")
    if vae is None:
        raise ValueError(f"Failed to load VAE from {args.wan_vae_path}")
        
    # Load DiT (Base for LoRA)
    dit_path = args.wan_dit_path
    if os.path.isdir(dit_path):
        # Support directory containing split safetensors
        files = sorted([str(p) for p in Path(dit_path).glob("*.safetensors")])
        if not files:
            raise ValueError(f"No .safetensors found in directory: {dit_path}")
        print(f"Detected directory for DiT. Loading {len(files)} files.")
        dit_path = files
        
    pool.auto_load_model(dit_path)
    dit = pool.fetch_model("wan_video_dit")
    if dit is None:
        raise ValueError(f"Failed to load DiT from {args.wan_dit_path}")
        
    # Wrap in DelitWanLoRA
    model = DelitWanLoRA(vae, dit, lora_rank=args.lora_rank)
    model.to(device)
    
    # 2. Dataset
    dataset = FaceOLATDelitDataset(
        data_root=args.data_root,
        env_resolution=(512, 1024),
        image_size=512,
        use_mask=True
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # 3. Optimizer
    # Only optimize LoRA parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr)
    
    # 4. Scheduler (Flow Matching)
    scheduler = FlowMatchScheduler(template="Wan")
    scheduler.set_timesteps(num_inference_steps=1000, training=True)
    
    # 5. Training Loop
    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    
    # Mixed Precision info
    dtype = torch.float32
    if args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        dtype = torch.float16
        
    print(f"Starting training with {len(dataset)} samples...")
    
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            # Move to device and cast dtype
            relit_equir = batch['relit_equir'].to(device).to(dtype)
            person_mask = batch['person_mask_equir'].to(device).to(dtype)
            valid_mask = batch['valid_mask'].to(device).to(dtype)
            env_encoded = batch['env_encoded'].to(device).to(dtype)
            
            with torch.cuda.amp.autocast(enabled=args.mixed_precision != "no", dtype=dtype):
                # 1. VAE Encode Target Env Map (Latent Target)
                # We need to use "encode_latents" but carefully.
                # env_encoded is 4 channels. 
                # Our VAE wrapper currently mimics `encode_latents` by taking first 3 channels.
                # Let's trust that for now (or improve logic if implemented differently).
                with torch.no_grad():
                    z_target = model.encode_latents(env_encoded) # [B, 16, 1, h, w]
                    
                # 2. Sample Noise & Timestep
                noise = torch.randn_like(z_target)
                timestep = torch.randint(0, 1000, (z_target.shape[0],), device=device)
                
                # 3. Add Noise (Flow Matching)
                # Scheduler handles broadcasting?
                # Scheduler expects sample, noise, timestep.
                # Timestep logic in scheduler: argmin(abs(timesteps - t)).
                # We need to map integer timestep indices to scheduler values?
                # scheduler.set_timesteps(1000, training=True) creates self.timesteps (1000 values from 1.0 to 0.0)
                # We picked integer t.
                # In `training_weight`, it calls argmin.
                # Let's pass the float timestep from scheduler.timesteps
                
                # Correct usage:
                # timesteps_float = scheduler.timesteps[timestep_indices]
                # But scheduler.timesteps is a tensor.
                # Let's just pass `scheduler.timesteps[timestep]`
                
                batch_timesteps = scheduler.timesteps[timestep.cpu()].to(device)
                
                # Add noise: (1-t)*x + t*noise? Check `add_noise` code: (1-sigma)*x + sigma*noise.
                # Wait, Flow Matching is x1 = (1-t)x0 + t*x1?
                # Wan uses Rectified Flow?
                # `scheduler.add_noise` uses `sigma`.
                z_t = scheduler.add_noise(z_target, noise, batch_timesteps)
                
                # 4. Forward
                # Predict Noise? Or Velocity?
                # Flow Matching target: v = x1 - x0 = noise - data.
                # `scheduler.training_target` returns `noise - sample` (if sample is data).
                target = scheduler.training_target(z_target, noise, batch_timesteps)
                
                pred = model(
                    relit_equir=relit_equir,
                    person_mask_equir=person_mask,
                    valid_mask=valid_mask,
                    noisy_latents=z_t,
                    timestep=batch_timesteps
                )
                
                # 5. Loss
                loss = F.mse_loss(pred, target)
                
            # Backward
            loss.backward()
            
            if (global_step + 1) % args.accumulate_grad == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            global_step += 1
            progress_bar.set_postfix(loss=loss.item())
            
            # Save
            if global_step % args.save_interval == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint_{global_step}.pt")
                model.save_lora(save_path)
        
        # Visualize
        visualize_inference(model, vae, scheduler, dataset, args, epoch, device, dtype)
                
    # Final save
    model.save_lora(os.path.join(args.output_dir, "last.pt"))
    print("Training finished.")

def visualize_inference(model, vae, scheduler, dataset, args, epoch, device, dtype):
    """
    Run inference on a single sample for visualization.
    """
    model.eval()
    save_dir = os.path.join(args.output_dir, "vis")
    os.makedirs(save_dir, exist_ok=True)
    
    # Pick first sample from dataset
    # We use the dataset directly to get CPU tensors
    # Ideally reuse validation set, but for now just use index 0 of train set
    sample = dataset[0] 
    
    relit_equir = sample['relit_equir'].unsqueeze(0).to(device).to(dtype) # [1, 4, H, W]
    person_mask = sample['person_mask_equir'].unsqueeze(0).to(device).to(dtype)
    valid_mask = sample['valid_mask'].unsqueeze(0).to(device).to(dtype)
    # env_encoded target for reference? 
    # Let's verify generation, so we don't need target ideally, but good for comparison.
    
    # Encode Condition
    with torch.no_grad():
        # Latent shape: [1, 16, 1, h, w]
        # Get shape from model's internal usage or just run forward part?
        # Model enc_latents returns [1, 16, 1, h, w]
        z_cond = model.encode_latents(relit_equir)
        
        # Initialize Noise (Latent)
        # Shape: same as z_cond
        latents = torch.randn_like(z_cond)
        
        # Sampling Loop
        # Use scheduler
        scheduler.set_timesteps(num_inference_steps=20, training=False)
        
        for t in tqdm(scheduler.timesteps, desc="Sampling Vis"):
            # Expand t for batch
            t_batch = torch.tensor([t], device=device).to(dtype)
            
            # Predict velocity/noise
            # Model forward: (relit, mask, valid, noisy_latents, timestep)
            noise_pred = model(
                relit_equir=relit_equir,
                person_mask_equir=person_mask,
                valid_mask=valid_mask,
                noisy_latents=latents,
                timestep=t_batch
            )
            
            # Step
            latents = scheduler.step(noise_pred, t_batch, latents)
            
        # Decode Latents to Image
        # VAE Decode: [B, 16, T, h, w] -> [B, 3, T, H, W]
        # Wan VAE decode expects [B, 16, T, h, w]
        decoded_3ch = vae.model.decode(latents.float()) # VAE usually FP32?
        # Output [B, 3, T, H, W]
        # Squeeze T
        decoded_3ch = decoded_3ch.squeeze(2) # [1, 3, H, W]
        
        # This is strictly "Normalized RGB" (or whatever VAE output is).
        # We need to construct 4th channel (Log Lum)? 
        # Wait, our model was trained to predict latent of "Encoded 4ch Env Map"? 
        # No, check `encode_latents` in delit_model_lora.py.
        # It takes 4ch input, selects first 3 channels (RGB), and encodes to latent.
        # So we only trained on RGB structure!
        # The 4th channel INTENSITY was passed via `mask_cond` to DiT, but NOT effectively reconstructed by VAE 
        # because VAE only outputs 3 channels.
        
        # This confirms Reconstruction Loss is HARD because we don't output HDR intensity freely.
        # We only output RGB.
        # However, for visualization, let's just save the RGB output.
        
        img_np = decoded_3ch[0].permute(1, 2, 0).detach().cpu().numpy() # [H, W, 3]
        img_np = np.clip(img_np, 0, 1)
        
        # Helper to save
        save_path = os.path.join(save_dir, f"epoch_{epoch+1}_vis.png")
        # RGB to BGR for cv2
        img_bgr = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_bgr)
        print(f"Saved visualization to {save_path}")



if __name__ == "__main__":
    args = parse_args()
    train(args)
