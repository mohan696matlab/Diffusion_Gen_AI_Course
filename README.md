
# Diffusion Models Course Repository

Welcome to a hands-on course repository covering diffusion models — from basic implementations to advanced applications like image editing, text conditioning, ControlNet, and adapter-based fine-tuning. This repository accompanies a YouTube series on diffusion models. [YouTube Series](https://www.youtube.com/playlist?list=PLoSULBSCtofearln-pGND44nr69FE9eIM)

## 📁 Repository Structure

### Part 1: Simple Diffusion
- Introduction to basic diffusion model concepts 
- Simple implementation examples
- Introduction to Diffusion based Generative MOdel [YouTube Video 1](https://youtu.be/QrZ7u29ITtw)
- Conditional Generation [YouTube Video 2](https://youtu.be/OE3KFv1zyUs)

### Part 2: MNIST Diffusion
- Application of diffusion models on MNIST dataset
- Training and evaluation scripts
- Using pure Pytorch [YouTube Video 3](https://youtu.be/Zm1MekFAjto)
- Using the Diffusers Library [YouTube Video 4](https://youtu.be/_dgp2q-YyOQ)

### Part 3: Celeb Faces Diffusion
- Advanced diffusion model implementation
- Celebrity face generation using TMDB dataset
- Key features:
  - Custom data preprocessing and filtering
  - Optimized training pipeline with gradient accumulation
  - Configurable evaluation steps
  - GPU acceleration support
- Finetuning a Diffusion model from scratch on Celebrity faces [YouTube Video 5](https://youtu.be/05yjbi-ySR4)

### Part 4: Image Editing with Diffusion [YouTube Video 6](https://youtu.be/RwgzDtmSC5g)
- Image manipulation using diffusion models
- Practical editing applications

### Part 5: Latent Diffusion Model (LDM)
- Implementation of latent diffusion models
- Celebrity face generation in latent space
- Enhanced features:
  - VAE integration for efficient training
  - Larger dataset support (1000 samples)
  - Optimized learning rate (5e-4)
  - Extended evaluation intervals

### Part 6: Stable Diffusion Text Conditioning
- Text-to-image generation capabilities
- Integration with text prompts

### Part 7: ControlNet
- Advanced control mechanisms for diffusion models
- Fine-grained generation control

### Part 8: IP-Adapters / Image Prompt-adapters
- Adapter-based conditioning (IPAdapter, face/ipadapter examples)
- Notebooks and examples showing how to combine adapters with ControlNet

### Part 9: LoRA Fine-tuning
- Low-Rank Adaptation (LoRA) examples and loading fine-tuned LoRA weights
- Notebooks demonstrating LoRA integration with Stable Diffusion

## 🛠️ Technical Stack

- PyTorch
- Hugging Face Diffusers
- Custom UNet2DModel implementations
- DDPMScheduler for noise scheduling
- Inpainting and Image to Image diffusion
- Controlnet and IP-Adapters
- LoRA (Prameter efficient finetuning)
- Optional CUDA / GPU acceleration


## 🎯 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/mohan696matlab/Diffusion_Gen_AI_Course.git
cd Diffusion_Gen_AI_Course
```

### 2. Create and activate a Python environment
You can use a `venv` or `conda` environment. Example `venv` on Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```
Or using `cmd.exe`:
```cmd
venv\Scripts\activate.bat
```
If you prefer `conda`:
```powershell
conda create -n diffusion python=3.11 -y
conda activate diffusion
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Try a notebook or script
- Open the notebooks in each `part_*` directory (for example `part_2_mnist/part_2_1_diffusion_from_scratch_pytorch.ipynb`).
- Example training or run scripts are present in parts that require them, e.g. `part_3_diffusion_celeb_faces/train_celeb_faces.py` and `part_5_latent_diffusion_model/train_ldm_celeb_faces.py`.

Each part can be run independently; check the corresponding notebook or README inside each part for dataset and runtime specifics.



Each part contains its own training configuration and can be run independently.

This repository provides a practical learning path for implementing diffusion models, from fundamental concepts to advanced conditioning and fine-tuning techniques. Use the notebooks for step-by-step walkthroughs and the scripts for experiments at scale.
