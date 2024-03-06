import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from ddpm import DDPM
from unet import Simple_Unet
from dataset import DDPM_Dataset
from tqdm import tqdm

data_path = "~/ダウンロード/CelebA/Img/img_align_celeba/"
data_path = os.path.expanduser(data_path)
img_files = os.listdir(data_path)

epochs = 1
batch_size = 64
noise_steps = 1000
learning_rate = 1e-4

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = DDPM_Dataset(data_path=data_path, img_files=img_files, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

unet = Simple_Unet(in_channels=3, out_channels=3)
if torch.cuda.device_count() > 1:
    unet = nn.DataParallel(unet, device_ids=[0, 1])
    # model = nn.DataParallel(model)

model = DDPM(denoise_network=unet, noise_steps=noise_steps, beta_start=0.0001, beta_end=0.02, device=device)
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_losses = []
for epoch in range(epochs):
    print("epoch {}:".format(str(epoch+1)))
    unet.to(device)
    model.train()
    with tqdm(train_dataloader, total=len(train_dataloader)) as pbar:
        for i, imgs in enumerate(pbar):
            imgs = imgs.to(device)
            time_steps = torch.randint(0, noise_steps, (batch_size,)).long().to(device)
            noisy_imgs, gt_noise = model.add_noise(imgs, time_steps)

            pred_noise = model.reverse_process(noisy_imgs, time_steps)

            loss = criterion(pred_noise, gt_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach().item())
    
    print("mean_train_loss:{}".format(sum(train_losses)/len(train_losses)))
    unet.to("cpu")
    torch.save(unet.module.state_dict(), "./checkpoint/DDPM_SimpleUnet.pth")
    print("save model")