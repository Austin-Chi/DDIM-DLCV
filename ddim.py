import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from utils import beta_scheduler
from UNet import UNet
from PIL import Image
import torchvision.transforms as T
import numpy as np
import sys

class DDIM:
    def __init__(self, unet_path, n_timestep=1000, ddim_timesteps=50, eta=0.0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_timestep = n_timestep
        self.ddim_timesteps = ddim_timesteps
        self.eta = eta

        # Load the pretrained UNet
        self.unet = UNet()
        self.unet.load_state_dict(torch.load(unet_path))
        self.unet.to(self.device)
        self.unet.eval()

        # Define betas and calculate related terms
        self.betas = beta_scheduler(n_timestep)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])

        # Create time step schedule for DDIM (50 steps uniform)
        self.ddim_timesteps_seq = np.asarray(list(range(0, self.n_timestep, self.n_timestep // self.ddim_timesteps)))
        self.ddim_timesteps_seq = self.ddim_timesteps_seq + 1
        self.ddim_timesteps_seq_prev = np.append(np.array([0]), self.ddim_timesteps_seq[:-1])

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def load_all_noise(self, noise_folder):
        noise_files = sorted([f for f in os.listdir(noise_folder) if f.endswith('.pt')])
        noise_batch = [torch.load(os.path.join(noise_folder, f)) for f in noise_files]
        return torch.cat(noise_batch, dim=0), noise_files

    def sample(self, noise_batch, clip_denoised=True):
        # Predict the noise residual
        with torch.no_grad():
            x = noise_batch  # Batch of noise
            batch_size = x.size(0)
            # print(x.size())
            device = next(self.unet.parameters()).device
            for i in reversed(range(0, self.ddim_timesteps)):
                t = torch.full((batch_size,), self.ddim_timesteps_seq[i], device=device, dtype=torch.long)
                t_prev = torch.full((batch_size,), self.ddim_timesteps_seq_prev[i], device=device, dtype=torch.long)


                
                pred_noise = self.unet(x, t)

                # DDIM sampling formula
                alpha_t = self._extract(self.alphas_cumprod, t, noise_batch.shape)
                alpha_t_prev = self._extract(self.alphas_cumprod, t_prev, noise_batch.shape)

                # Predicted x0
                pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)

                if clip_denoised:
                    pred_x0 = torch.clamp(pred_x0, min=-1.0, max=1.0)

                # calculate x_{t-1}
                sigma_t = self.eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
                # Direction pointing to x_t
                dir_xt = (1 - alpha_t_prev - sigma_t**2).sqrt() * pred_noise

                # Stochastic path
                noise = torch.randn_like(x)

                x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma_t * noise

        return x.cpu()

    def evaluate_mse(self, generated_imgs, ground_truth_folder):
        transform = T.ToTensor()
        mse_loss = 0
        for idx, img in enumerate(generated_imgs):
            # Load ground truth image
            gt_img_path = os.path.join(ground_truth_folder, f"{idx:02d}.png")
            gt_img = Image.open(gt_img_path)
            gt_img = transform(gt_img).to(img.device)

            # Calculate MSE
            mse_loss += F.mse_loss(img, gt_img).item()
        
        mse_loss /= len(generated_imgs)  # Average MSE
        return mse_loss

# Usage example
if __name__ == "__main__":
    # Define paths
    noise_folder = sys.argv[1]#"hw2_data/face/noise"
    unet_path = sys.argv[3]#"hw2_data/face/UNet.pt"
    save_path = sys.argv[2]#"p2/outputs"
    # gt_folder = "hw2_data/face/GT"

    # Create the output directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Initialize DDIM model
    ddim = DDIM(unet_path=unet_path, n_timestep=1000, ddim_timesteps=50, eta=0.0)

    # Load all noise files as a batch
    noise_batch, noise_files = ddim.load_all_noise(noise_folder)

    # Generate images from all noise files
    generated_imgs = ddim.sample(noise_batch)

    # Save generated images
    for idx, img in enumerate(generated_imgs):
        with torch.no_grad():
            img_min = torch.min(img)
            img_max = torch.max(img)

            #normalize
            img = (img - img_min) / (img_max - img_min)
            img = torch.clamp(img, min=-1, max=1)
        save_image(img, f"{save_path}/{idx:02d}.png")
        generated_imgs[idx] = img

    # Evaluate MSE against ground truth
    # mse_score = ddim.evaluate_mse(generated_imgs, gt_folder)
    # print(f"Average MSE: {mse_score}")
