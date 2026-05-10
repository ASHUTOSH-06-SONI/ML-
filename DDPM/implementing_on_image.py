import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Load image
img = Image.open("billi.jpeg").convert("RGB")

# Convert to tensor
transform = transforms.ToTensor()
x0 = transform(img)

# Generate noise
noise = torch.randn_like(x0)

beta = 1.0

# Forward noising
xt = (
    torch.sqrt(torch.tensor(1 - beta)) * x0
    + torch.sqrt(torch.tensor(beta)) * noise
)

# Clamp values
xt = torch.clamp(xt, 0, 1)

# Create side-by-side plot
fig, axes = plt.subplots(1, 2, figsize=(10,5))

# Original
axes[0].imshow(x0.permute(1,2,0))
axes[0].set_title("Original Image")
axes[0].axis("off")

# Noisy
axes[1].imshow(xt.permute(1,2,0))
axes[1].set_title("Noisy Image")
axes[1].axis("off")

plt.show()