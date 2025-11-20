
# Diffusion Models Course Repository

Welcome to a comprehensive course repository covering various aspects of diffusion models, from basic implementation to advanced applications like image editing and text conditioning. This repository is aprt of the youtube video serise on diffusion models.[Link: YouTube Series](https://www.youtube.com/playlist?list=PLoSULBSCtofearln-pGND44nr69FE9eIM)

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

## 🛠️ Technical Stack

- PyTorch
- Diffusers Library
- Custom UNet2DModel implementations
- DDPMScheduler for noise scheduling
- AdamW optimizer


## 🎯 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/mohan696matlab/Diffusion_Gen_AI_Course.git
```
### 2. Create a new virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install required packages
```bash
pip install -r requirements.txt
```



Each part contains its own training configuration and can be run independently.

This repository provides a complete learning path for understanding and implementing diffusion models, from basic concepts to advanced applications. Each part builds upon previous knowledge while introducing new concepts and techniques.