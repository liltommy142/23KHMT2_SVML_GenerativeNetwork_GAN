import argparse
import os
import numpy as np
import math
import json

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

# Tạo các thư mục cần thiết
os.makedirs("images", exist_ok=True)  # Thư mục lưu hình ảnh sample
os.makedirs("saved_models", exist_ok=True)  # Thư mục lưu model weights
os.makedirs("logs", exist_ok=True)  # Thư mục lưu log và loss history

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints (số epoch giữa mỗi lần lưu model). Đặt -1 để tắt tính năng lưu định kỳ")

opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
if cuda:
    print(f"Using device: CUDA/GPU ({torch.cuda.get_device_name(0)})")
else:
    print("Using device: CPU")


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Khởi tạo các list để lưu loss history (dùng để vẽ đồ thị)
# Lưu loss trung bình cho mỗi epoch
epoch_d_losses = []  # Discriminator loss theo epoch
epoch_g_losses = []  # Generator loss theo epoch
epoch_numbers = []   # Số epoch tương ứng

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    # Khởi tạo list để lưu loss của tất cả batches trong epoch hiện tại
    # Sau đó sẽ tính trung bình để có loss của epoch
    batch_d_losses = []
    batch_g_losses = []

    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = torch.ones(imgs.size(0), 1, device=imgs.device).type(Tensor)
        fake = torch.zeros(imgs.size(0), 1, device=imgs.device).type(Tensor)
        valid = Variable(valid, requires_grad=False)
        fake = Variable(fake, requires_grad=False)

        # Configure input
        real_imgs = imgs.type(Tensor)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(
            0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # Lưu loss của batch hiện tại vào list
        # item() để chuyển từ tensor về số Python thuần
        batch_d_losses.append(d_loss.item())
        batch_g_losses.append(g_loss.item())

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        # Lưu hình ảnh sample theo định kỳ
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" %
                       batches_done, nrow=5, normalize=True)

    # Tính loss trung bình cho epoch hiện tại
    avg_d_loss = np.mean(batch_d_losses)
    avg_g_loss = np.mean(batch_g_losses)
    
    # Lưu loss trung bình của epoch vào list
    epoch_d_losses.append(avg_d_loss)
    epoch_g_losses.append(avg_g_loss)
    epoch_numbers.append(epoch + 1)
    
    # In loss trung bình của epoch
    print(f"[Epoch {epoch + 1}/{opt.n_epochs}] Average losses - D: {avg_d_loss:.4f}, G: {avg_g_loss:.4f}")

    # Lưu model checkpoint theo định kỳ (sau mỗi N epochs)
    # checkpoint_interval = -1 nghĩa là tắt tính năng lưu định kỳ
    if opt.checkpoint_interval != -1 and (epoch + 1) % opt.checkpoint_interval == 0:
        # Lưu state_dict của generator và discriminator
        # state_dict chứa tất cả các tham số (weights và biases) của model
        torch.save(generator.state_dict(),
                   "saved_models/generator_%d.pth" % (epoch + 1))
        torch.save(discriminator.state_dict(),
                   "saved_models/discriminator_%d.pth" % (epoch + 1))
        print(f"[Checkpoint] Đã lưu model tại epoch {epoch + 1}")

# Lưu model cuối cùng sau khi train xong tất cả epochs
print("Training hoàn tất! Đang lưu model cuối cùng...")
torch.save(generator.state_dict(), "saved_models/generator_final.pth")
torch.save(discriminator.state_dict(), "saved_models/discriminator_final.pth")
print("[Final Checkpoint] Đã lưu model cuối cùng vào saved_models/")
print("  - Generator: saved_models/generator_final.pth")
print("  - Discriminator: saved_models/discriminator_final.pth")

# Lưu loss history vào file JSON để vẽ đồ thị sau này
# Format: {epoch: [d_loss, g_loss]}
loss_history = {
    "epochs": epoch_numbers,
    "discriminator_losses": epoch_d_losses,
    "generator_losses": epoch_g_losses
}

# Lưu vào file JSON
loss_file = "logs/loss_history.json"
with open(loss_file, "w") as f:
    json.dump(loss_history, f, indent=2)

print(f"[Loss History] Đã lưu loss history vào {loss_file}")
print(f"  - Tổng số epochs: {len(epoch_numbers)}")
print(f"  - D loss cuối cùng: {epoch_d_losses[-1]:.4f}")
print(f"  - G loss cuối cùng: {epoch_g_losses[-1]:.4f}")
