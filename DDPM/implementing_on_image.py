import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# PART 1- X0 SE XT, WE JUST ADD NOISE

img = Image.open("billi.jpeg").convert("RGB")
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
x0 = transform(img)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
x0 = x0.to(device)

T = 1000
betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

t = torch.randint(0, T, (1,), device=device)
noise = torch.randn_like(x0)

xt = torch.sqrt(alpha_bar[t]) * x0 + torch.sqrt(1 - alpha_bar[t]) * noise

xt_display = torch.clamp(xt, 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(x0.detach().cpu().permute(1, 2, 0))
axes[0].set_title("Original Image")
axes[0].axis("off")

# Noisy

axes[1].imshow(xt_display.detach().cpu().permute(1, 2, 0))
axes[1].set_title("Noisy Image")
axes[1].axis("off")


#THEN THIS GOES PAST UNET TO "PREDICT" NOISE

import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):

        half = self.dim // 2

        emb = torch.log(torch.tensor(10000.0, device=t.device)) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        return emb


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.time_proj = nn.Linear(time_dim, out_channels)

    def forward(self, x, t_emb):

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x)

        time = self.time_proj(t_emb)
        time = time[:, :, None, None]

        x = x + time

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x)

        return x


class UNet(nn.Module):

    def __init__(self, in_channels=3, time_dim=128):
        super().__init__()

        self.time_embedding = TimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))

        # Encoder

        self.down1 = ConvBlock(3, 64, time_dim)
        self.down2 = ConvBlock(64, 128, time_dim)

        # Bottleneck

        self.mid = ConvBlock(128, 256, time_dim)

        # Decoder

        self.up2 = ConvBlock(256 + 128, 128, time_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_dim)

        self.output = nn.Conv2d(64, 3, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):

        # Time embedding

        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)

        # Encoder

        x1 = self.down1(x, t_emb)
        x2 = self.pool(x1)
        x2 = self.down2(x2, t_emb)

        # Bottleneck

        x3 = self.pool(x2)
        x3 = self.mid(x3, t_emb)

        # Decoder

        x3 = F.interpolate(x3, size=x2.shape[-2:], mode="nearest")
        x3 = torch.cat([x3, x2], dim=1)
        x3 = self.up2(x3, t_emb)

        x3 = F.interpolate(x3, size=x1.shape[-2:], mode="nearest")
        x3 = torch.cat([x3, x1], dim=1)
        x3 = self.up1(x3, t_emb)

        return self.output(x3)


# PART 3- REVERSE EQUATION

print("Device:", device)

model = UNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 10000

for step in range(epochs):

    t = torch.randint(0, T, (1,), device=device)
    noise = torch.randn_like(x0)

    xt = torch.sqrt(alpha_bar[t]) * x0 + torch.sqrt(1 - alpha_bar[t]) * noise

    pred_noise = model(xt.unsqueeze(0), t)

    loss = F.mse_loss(pred_noise, noise.unsqueeze(0))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print("Step:", step, "Loss:", loss.item())

torch.save(model.state_dict(), "ddpm_model.pth")


model.eval()

t = torch.tensor([500], device=device)
noise = torch.randn_like(x0)

xt = torch.sqrt(alpha_bar[t]) * x0 + torch.sqrt(1 - alpha_bar[t]) * noise

with torch.no_grad():

    pred_noise = model(xt.unsqueeze(0), t)

    predicted_x0 = (xt.unsqueeze(0) - torch.sqrt(1 - alpha_bar[t]) * pred_noise) / torch.sqrt(alpha_bar[t])

predicted_x0 = torch.clamp(predicted_x0.squeeze(0), 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(x0.cpu().permute(1, 2, 0))
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(torch.clamp(xt, 0, 1).cpu().permute(1, 2, 0))
axes[1].set_title("Noisy Image")
axes[1].axis("off")

axes[2].imshow(predicted_x0.cpu().permute(1, 2, 0))
axes[2].set_title("Predicted x0")
axes[2].axis("off")

plt.show()


# PART 4- REVERSE DIFFUSION

model.eval()

x = torch.randn_like(x0).unsqueeze(0)

with torch.no_grad():

    for t in reversed(range(T)):

        t_tensor = torch.tensor([t], device=device)

        pred_noise = model(x, t_tensor)

        alpha_t = alphas[t]
        alpha_bar_t = alpha_bar[t]
        beta_t = betas[t]

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise
        ) + torch.sqrt(beta_t) * noise

x = x.squeeze(0)
x = torch.clamp(x, 0, 1)

plt.figure(figsize=(5, 5))
plt.imshow(x.permute(1, 2, 0).cpu())
plt.title("Generated Image")
plt.axis("off")
plt.show()