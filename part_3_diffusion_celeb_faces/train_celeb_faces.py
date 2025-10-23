import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from IPython.display import clear_output
import time
import seaborn as sns
from tqdm.auto import tqdm
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import torch
from diffusers import DDPMScheduler
from diffusers import UNet2DModel
from diffusers import DDIMScheduler
import os
import shutil

shutil.rmtree('runs', ignore_errors=True)
os.makedirs("runs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')



ds = load_dataset("ashraq/tmdb-celeb-10k",split='train')
ds = ds.select(range(500))
ds = ds.filter(lambda x: x['gender'] in [1, 2])



transform = transforms.Compose([ transforms.Resize((64,64)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5],[0.5]),
])

def transforms(examples):
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
    return examples

ds = ds.with_transform(transforms)



def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example["pixel_values"]))
        labels.append(example["gender"])
        
    pixel_values = torch.stack(images)
    labels = torch.tensor(labels).long() - 1
    return {"pixel_values": pixel_values, "label":labels}

dataloader = DataLoader(ds, collate_fn=collate_fn, batch_size=4)



scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
print(scheduler)



unet = UNet2DModel(
    sample_size=64,
    in_channels = 3,
    out_channels = 3,
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    
    down_block_types = ('DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D'),
    mid_block_type = 'UNetMidBlock2D',
    up_block_types = ('AttnUpBlock2D', 'AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D'),
    block_out_channels= (64, 128, 160, 224),
    num_class_embeds=2
)

unet.to(device)



def generate_image():
    torch.manual_seed(42)
    os.makedirs("runs", exist_ok=True)
    xt = torch.randn(4,3,64,64).to(device)
    label = torch.tensor([0,0,1,1]).long().to(device)
    
    inf_scheduler = DDIMScheduler.from_config(scheduler.config)
    inf_scheduler.set_timesteps(50)

    unet.eval()

    for t in tqdm(inf_scheduler.timesteps,total=inf_scheduler.num_inference_steps,leave=False):
        with torch.no_grad():
            noise_pred = unet(xt,t.to(device), label).sample
            
        
        xt = inf_scheduler.step(noise_pred, t.to(device), xt).prev_sample
        
    # Save image at each step
    grid = torchvision.utils.make_grid(xt * 0.5 + 0.5, nrow=2)
    img = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(img).save(f"runs/gen_image_step_{num_steps:03d}.png")
    
    

# Our loss function
loss_fn = nn.MSELoss()

# The optimizer
opt = torch.optim.AdamW(unet.parameters(), lr=1e-4) 

# Keeping a record of the losses for later viewing
losses = []

num_steps = 0
accumulation_steps = 4  # accumulate gradients over 4 mini-batches
max_train_steps=50000
eval_steps=500


progress_bar = tqdm(
    range(0, max_train_steps),
    desc="Steps"
)


# The training loop
while num_steps < max_train_steps:
    for batch in dataloader:
        
        # Get some data and prepare the corrupted version
        x = batch['pixel_values'].to(device) 
        y = batch['label'].long().to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
        noisy_x = scheduler.add_noise(x, noise, timesteps)

        # Get the model prediction
        pred = unet(noisy_x, timesteps, y).sample # Note that we pass in the labels y

        # Calculate the loss
        loss = loss_fn(pred, noise) / accumulation_steps  # scale loss # How close is the output to the noise
        # Backward pass (accumulate gradients)
        loss.backward()

        # Update parameters every `accumulation_steps` mini-batches
        if (num_steps + 1) % accumulation_steps == 0:
            opt.step()
            opt.zero_grad()
        
        num_steps += 1

        # Store the loss for later
        losses.append(loss.item())
        
        # Update progress bar with loss value
        progress_bar.set_postfix(loss=loss.item())  # Assuming loss is a tensor; use .item() to get the float value
        progress_bar.update(1)
    
        if num_steps % eval_steps == 0:
            avg_loss = sum(losses[-100:])/100
            print(f'num_steps: {num_steps}, loss: {avg_loss:05f}')
            generate_image()
        
        if num_steps >= max_train_steps:
            break


        