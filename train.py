from models import Generator, Discriminator

import os
import torch
import torchvision
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms import Lambda
from torch.utils.data.dataloader import DataLoader

import matplotlib.pyplot as plt 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GAN training step
def train_epoch(epoch, gen, disc, optim_g, optim_d, loader):
    gen_losses, disc_losses = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        batch_size = imgs.shape[0]

        # Train discriminator
        noise = torch.randn(batch_size, 3, 36, 36).to(device)
        gen.eval()
        with torch.no_grad():
            fake_imgs = gen(noise, labels)

        disc.train()
        disc_r = disc(imgs).softmax(-1)
        disc_f = disc(fake_imgs).softmax(-1)

        loss_disc_r = F.cross_entropy(disc_r, labels)
        loss_disc_f_entropy = -torch.sum(disc_f * torch.log(disc_f), dim=-1)

        # minimize the cross entropy with true labels, maximize entropy with fake input
        loss_disc = loss_disc_r - loss_disc_f_entropy
        optim_d.zero_grad()
        loss_disc.backward()
        optim_d.step()
        disc_losses += loss_disc.item()

        # Train generator
        gen.train()
        noise = torch.randn(batch_size, 3, 36, 36).to(device)
        gen_data = gen(noise, labels)
        disc_gen = disc(gen_data)
        loss_gen = F.cross_entropy(disc_gen, labels)
        optim_g.zero_grad()
        loss_gen.backward()
        optim_g.step()
        gen_losses += loss_gen.item()

    evaluate(gen, imgs, labels, str(epoch))
    return gen_losses/len(loader), disc_losses/len(loader)

@torch.no_grad()
def evaluate(gen, imgs, labels, name):
    reverse_transform = transforms.Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    gen.eval()
    imgs, labels = imgs[:4].to(device), labels[:4].to(device)
    batch_size = imgs.shape[0]

    noise = torch.randn(batch_size, 3, 36, 36).to(device)
    gen_imgs = gen(noise, labels)

    real_imgs = [reverse_transform(img.cpu()) for img in imgs]
    gen_imgs = [reverse_transform(img.cpu()) for img in gen_imgs]

    fig, axs = plt.subplots(figsize=(200,200), nrows=len(real_imgs), ncols=2, squeeze=False)
    for row_idx in range(len(real_imgs)):
        ax = axs[row_idx, 0]
        ax.imshow(real_imgs[row_idx])
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        ax = axs[row_idx, 1]
        ax.imshow(gen_imgs[row_idx])
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()
    plt.savefig(f"{name}.png")


import argparse
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("datafolder", default="data", nargs="?", help="The data folder")    
    parser.add_argument("--seed", type=int, default=42, help="The random seed")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print(f"args: {args}")
    
    torch.manual_seed(args.seed)

    image_size = 128
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),          # CHW, values in [0, 1]
        Lambda(lambda t: (t * 2) - 1),  # rescale to [-1, 1]
    ])

    dataset = ImageFolder(args.datafolder, transform=transform)
    num_classes = len(dataset.classes)
    print(f"dataset size: {len(dataset)}")

    batch_size = args.batch_size
    num_workers = 4 if torch.cuda.is_available() else 0
    train_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    gen = Generator(num_classes).to(device)
    disc = Discriminator(num_classes).to(device)

    optim_g = torch.optim.Adam(gen.parameters(), lr=1e-5)
    optim_d = torch.optim.Adam(disc.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        loss_g, loss_d = train_epoch(epoch, gen, disc, optim_g, optim_d, loader)
        print(f"[{epoch}] g:{loss_g:.4f}, d_loss:{loss_d:.4f}")
        
    
    torch.save(gen.state_dict(), "state_dict/gen.pth")
    torch.save(disc.state_dict(), "state_dict/disc.pth")
    torch.save(optim_g.state_dict(), "state_dict/optim_g.pth")
    torch.save(optim_d.state_dict(), "state_dict/optim_d.pth")
    print("Done.")

# import torch
# import random
# import numpy as np
# torch.manual_seed(args.seed)
# random.seed(args.seed)
# np.random.seed(args.seed)