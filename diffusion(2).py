import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from kornia.contrib import connected_components


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        # 编码器部分
        self.enc1 = self.conv_block(in_channels, 16)
        self.enc2 = self.conv_block(16, 32)
        # self.enc3 = self.conv_block(128, 256)
        # self.enc4 = self.conv_block(256, 512)
        self.center = self.center_block(32, 64)
        # # 最底部的卷积块
        # self.center = self.conv_block(512, 1024)

        # # 解码器部分
        # self.up4 = self.upconv_block(1024, 512)
        # self.up3 = self.upconv_block(512, 256)
        self.up2 = self.upconv_block(64, 32)
        self.up1 = self.upconv_block(32, 16)
        self.maxpool = nn.MaxPool3d(2)
        self.trans2 = self.conv_Transpose(64, 32)
        self.trans1 = self.conv_Transpose(32, 16)
        # 输出层
        self.dropout = nn.Dropout(0.1)
        self.out_conv = nn.Conv3d(16, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """卷积块：卷积 -> 激活 -> 卷积"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def conv_Transpose(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def center_block(self, in_channels, out_channels):
        """反卷积块：上采样 -> 卷积"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

        )

    def upconv_block(self, in_channels, out_channels):
        """反卷积块：上采样 -> 卷积"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器部分
        enc1 = self.enc1(x)
        enc1_m = self.maxpool(enc1)
        # print("enc1:",enc1.shape)
        enc2 = self.enc2(enc1_m)
        enc2_m = self.maxpool(enc2)
        # print("enc2_m:",enc2_m.shape)

        # enc3 = self.enc3(enc2)
        # enc4 = self.enc4(enc3)

        # 底部
        center = self.center(enc2_m)
        # print("center:",center.shape)
        center = self.trans2(center)

        # 解码器部分
        center = torch.cat([enc2, center], dim=1)
        up2 = self.up2(center)
        up2 = self.trans1(up2)
        # up4 = torch.cat([up4, enc4], dim=1)  # 跳跃连接
        # up3 = self.up3(up4)
        # up3 = torch.cat([up3, enc3], dim=1)  # 跳跃连接
        # up2 = self.up2(up3)
        up2 = torch.cat([up2, enc1], dim=1)  # 跳跃连接
        up1 = self.up1(up2)
        # up0 = self.up0(u)

        # 输出层
        out = self.out_conv(up1)
        # out = self.dropout(out)
        return out


# 生成高斯噪声
def add_noise(image, noise_level=0.4):
    noise = torch.randn_like(image) * noise_level
    return torch.clamp(image + noise, 0, 1)


def compute_centers(mask_labels):
    """计算每个连通区域的中心，假设 mask_labels 是 (1, 1, H, W) 形状"""
    labels = mask_labels.unique()
    labels = labels[labels != 0]  # 排除背景
    centers = []
    for label in labels:
        coords = (mask_labels == label).nonzero(as_tuple=False)  # (N, 4)
        coords_hw = coords[:, -2:]  # 取 H, W
        center = coords_hw.float().mean(dim=0)
        centers.append(center)
    return centers


batch_size = 4
# 训练扩散模型
loss_list = []


def train_diffusion_model(model, dataloader, epochs=80, noise_level=0.4, avg_dist=32, alpha_dist=1e-4):
    epoch_phy = 60
    optimizer = optim.Adam(model.parameters(), lr=0.0008)
    criterion = nn.MSELoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch_phy, eta_min=0.0003)

    for epoch in range(epochs):
        total_loss = 0
        if epoch == epoch_phy:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0003

        for images in dataloader:
            loss_dist = 0
            images = images.to(device)
            noisy_images = add_noise(images, noise_level).to(device)

            optimizer.zero_grad()
            denoised_images = model(noisy_images)

            if epoch >= epoch_phy:
                red = denoised_images[:, 0, :, :, :]  # R channel
                green = denoised_images[:, 1, :, :, :]
                blue = denoised_images[:, 2, :, :, :]

                red_mask = F.relu(red - 0.5 - green) * F.relu(red - 0.5 - blue)  # shape: [B, T, H, W]
                red_mask = red_mask / (red_mask.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[
                                           0] + 1e-6)  # normalize to [0,1]

                # grid shape: [1, 1, H, W] → expand to [B, T, H, W]
                B, T, H, W = red_mask.shape
                y_coords = torch.arange(H, device=red_mask.device)
                x_coords = torch.arange(W, device=red_mask.device)
                grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
                grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
                grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

                # 第一颗球（红色掩码）
                total_mass = red_mask.sum(dim=(-2, -1), keepdim=True) + 1e-6
                cx = (red_mask * grid_x).sum(dim=(-2, -1), keepdim=True) / total_mass
                cy = (red_mask * grid_y).sum(dim=(-2, -1), keepdim=True) / total_mass
                center1 = torch.cat([cx, cy], dim=-1)  # [B, T, 2]

                # 第二颗球（左右翻转红掩码再求一次中心）
                flipped_mask = red_mask.flip(-1)
                total_mass_flipped = flipped_mask.sum(dim=(-2, -1), keepdim=True) + 1e-6
                flipped_cx = (flipped_mask * grid_x).sum(dim=(-2, -1), keepdim=True) / total_mass_flipped
                flipped_cy = (flipped_mask * grid_y).sum(dim=(-2, -1), keepdim=True) / total_mass_flipped
                center2 = torch.cat([flipped_cx, flipped_cy], dim=-1)  # [B, T, 2]

                # 距离差异损失（逐帧欧几里得距离）
                dists = torch.norm(center1 - center2, dim=-1)  # [B, T]
                loss_dist = ((dists - avg_dist) ** 2).mean()

            loss = criterion(denoised_images, images) + alpha_dist * loss_dist
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        loss_list.append(total_loss)
        if epoch < epoch_phy:
            scheduler.step()

        # plt
        frame = noisy_images[0, :, 0, :, :].cpu().detach().numpy()
        frame = np.transpose(frame, (1, 2, 0))[:, :, ::-1]
        plt.figure(figsize=(5, 5))
        plt.imshow(frame)
        plt.title(f"Noisy Frame {1}")
        plt.axis('off')
        plt.show()

        frame = denoised_images[0, :, 0, :, :].cpu().detach().numpy()
        frame = np.transpose(frame, (1, 2, 0))[:, :, ::-1]
        plt.figure(figsize=(5, 5))
        plt.imshow(frame)
        plt.title(f"Frame {1}")
        plt.axis('off')
        plt.show()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")


class NPYVideoDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): 存放 .npy 文件的目录
            transform (callable, optional): 预处理转换
        """
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        video_data = np.load(file_path)  # 读取 .npy 文件, shape: (num_frames, H, W, C)

        # 归一化到 [0, 1]
        video_data = video_data.astype(np.float32)

        # 转换为 PyTorch 张量 (C, num_frames, H, W)
        video_data = torch.tensor(video_data).permute(3, 0, 1, 2)  # (C, num_frames, H, W)

        if self.transform:
            video_data = self.transform(video_data)

        return video_data


# 数据集路径
data_path = "input"

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    # transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1, 1]
])

# 创建 Dataset
dataset = NPYVideoDataset(data_path)
video = dataset[0]  # 选择第一个视频，假设返回形状为 (C, T, H, W)
# 你可以选择其他索引的视频, 比如 dataset[1], dataset[2] 等
print(video.shape)
# 假设 video 的形状是 (C, T, H, W)，我们可以选择其中的一些帧进行查看
C, T, H, W = video.shape

# # 显示第一个视频的前几帧
# for t in range(min(5, T)):  # 最多显示前5帧
#     frame = video[:, t, :, :].cpu().numpy()  # 取第 t 帧，形状为 (C, H, W)

#     # 转换为 H, W, C 形状
#     frame = np.transpose(frame, (1, 2, 0))  # (H, W, C)

#     # 显示图片
#     plt.figure(figsize=(5, 5))
#     plt.imshow(frame)
#     plt.title(f"Frame {t + 1}")
#     plt.axis('off')
#     plt.show()
# 创建 DataLoader
train_size = int(0.8 * len(dataset))
test_size = int(0.2 * len(dataset))

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 测试 DataLoader
for batch in train_loader:
    print(batch.shape)  # 预期输出: (batch_size, C, num_frames, H, W)
    break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练模型
from torchsummary import summary

# 假设你的模型是UNet类
model = UNet(in_channels=3, out_channels=3).to(device)

# 打印模型总结
summary(model, input_size=(3, 160, 200, 200))
model = UNet().to(device)
train_diffusion_model(model, train_loader, epochs=100, noise_level=0.4)

import cv2
import torch
import os
import numpy as np


def save_video_as_mp4(video_tensor, output_file, fps=160):
    """
    将视频数据保存为 .mp4 文件
    :param video_tensor: 视频张量，形状为 (C, T, H, W)
    :param output_file: 输出文件路径，保存为 .mp4 格式
    :param fps: 视频帧率（每秒帧数）
    """
    C, T, H, W = video_tensor.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 acv1 编码
    out = cv2.VideoWriter(output_file, fourcc, fps, (W, H))  # 输出视频尺寸应为 (W, H)

    for t in range(T):  # 遍历每一帧
        frame = video_tensor[:, t, :, :].cpu().numpy()  # (C, H, W)

        # 检查是否存在问题
        if np.max(frame) > 1:  # 如果最大值大于1，表示已经是 [0, 255] 范围内的数据
            print(f"Warning: Frame {t} pixel values are out of range [0, 1]")

        frame = np.transpose(frame, (1, 2, 0))  # 转换为 H, W, C 形状
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)  # 恢复到 [0, 255] 并转换为 uint8

        out.write(frame)  # 将帧写入视频

    out.release()  # 完成写入后释放


def generate_videos_from_tensor(data_tensor, output_dir='./output_videos',
                                fps=160):  # must be multiples of 4, because there are 2 downsampling
    """
    从输入的 tensor 中生成视频文件
    :param data_tensor: 输入数据张量，形状为 (B, C, T, H, W)，B 是批大小，C 是通道数，T 是帧数，H 和 W 是帧的高和宽
    :param output_dir: 保存视频的输出文件夹
    :param fps: 每秒帧数
    """
    os.makedirs(output_dir, exist_ok=True)  # 创建输出文件夹
    B, C, T, H, W = data_tensor.shape  # 获取数据张量的维度信息

    for video_idx in range(B):  # 遍历每个视频
        video_tensor = data_tensor[video_idx]  # 获取第 video_idx 个视频，形状为 (C, T, H, W)

        # 检查每个视频帧的大小
        print(f"Video {video_idx}: Shape of video tensor is {video_tensor.shape}")

        output_file = os.path.join(output_dir, f'video_{video_idx}.mp4')
        save_video_as_mp4(video_tensor, output_file, fps)
        print(f"Saved video {video_idx} to {output_file}")


# 测试去噪效果
def test_denoising(model, dataloader, noise_level=0.4):
    model.eval()
    with torch.no_grad():
        images = next(iter(dataloader))
        images = images.to(device)
        noisy_images = add_noise(images, noise_level).to(device)
        print(np.max(np.array(noisy_images.cpu())), np.min(np.array(noisy_images.cpu())))
        denoised_images = model(noisy_images.to(device))
        generate_videos_from_tensor(denoised_images, output_dir='./output_videos', fps=160)
        frame = images[0, :, 0, :, :].cpu().numpy()  # 取第 t 帧，形状为 (C, H, W)
        print(images.shape)

        # plt
        frame = denoised_images[0, :, 0, :, :].cpu().detach().numpy()
        frame = np.transpose(frame, (1, 2, 0))
        plt.figure(figsize=(5, 5))
        plt.imshow(frame)
        plt.title(f"Frame {1}")
        plt.axis('off')
        plt.show()


def test_denoising_with_pure_noise(model, dataloader, noise_level=0.4):
    model.eval()
    with torch.no_grad():
        images = next(iter(dataloader))  # 用来获取 shape
        images = images.to(device)

        # 生成纯噪音（与 images 尺寸相同）
        noise = torch.randn_like(images) * noise_level
        print("Pure noise range:", np.max(noise.cpu().numpy()), np.min(noise.cpu().numpy()))

        # 送入模型
        denoised_images = model(noise.to(device))

        # 保存视频
        generate_videos_from_tensor(denoised_images, output_dir='./output_videos', fps=160)

        # 可视化一帧
        frame = denoised_images[0, :, 0, :, :].cpu().detach().numpy()
        frame = np.transpose(frame, (1, 2, 0))  # (C, H, W) -> (H, W, C)

        plt.figure(figsize=(5, 5))
        plt.imshow(frame)
        plt.title("Generated from Pure Noise")
        plt.axis('off')
        plt.show()


test_denoising(model, test_loader)
# test_denoising_with_pure_noise(model, test_loader)

plt.figure(figsize=(10, 6))
plt.plot(loss_list)
plt.title(f"Loss")
plt.show()