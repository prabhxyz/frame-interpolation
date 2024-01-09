import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.nn.functional import interpolate as F_interpolate
from PIL import Image

class FrameInterpolationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(6, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.ReLU(inplace=True)
            ) for _ in range(3)
        ])
        self.resize = nn.Upsample(size=(90, 160), mode='bilinear', align_corners=False)
        self.fusion_conv = nn.Conv2d(384, 128, 3, 1, 1)
        self.upsample_conv = nn.ConvTranspose2d(128, 3, 3, 2, 1, 1, 1)

    def forward(self, x):
        feature_maps = [extractor(x) for extractor in self.feature_extractor]
        feature_maps_resized = [self.resize(fm) for fm in feature_maps]
        x = torch.cat(feature_maps_resized, 1)
        x = F.relu(self.fusion_conv(x))
        x = self.upsample_conv(x)
        return x

class FrameInterpolationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.video_list = [f"video{i}" for i in range(1, 26)]
        self.num_frames_per_video = 190

    def __len__(self):
        return len(self.video_list) * self.num_frames_per_video

    def __getitem__(self, idx):
        video_idx, frame_idx = divmod(idx, self.num_frames_per_video)
        frame_idx += 1
        video_folder = os.path.join(self.root_dir, f"video{video_idx + 1}")
        frame_idx = min(frame_idx, self.num_frames_per_video)
        frame1_path = os.path.join(video_folder, f"frame{frame_idx:04d}.jpg")
        frame2_path = os.path.join(video_folder, f"frame{frame_idx + 1:04d}.jpg")
        target_path = os.path.join(video_folder, f"frame{frame_idx + 2:04d}.jpg")
        frame1, frame2, target = map(Image.open, (frame1_path, frame2_path, target_path))
        if self.transform:
            frame1, frame2, target = map(self.transform, (frame1, frame2, target))
        return frame1, frame2, target

model = FrameInterpolationModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Loading dataset...")
dataset = FrameInterpolationDataset(root_dir='dataset/frames', transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print("Dataset loaded.")

print("Training model...")
num_epochs = 10
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    for frame1, frame2, target in dataloader:
        inputs = torch.cat((frame1, frame2), 1)
        optimizer.zero_grad()
        outputs = model(inputs)
        target_resized = F_interpolate(target, size=outputs.shape[2:], mode='bilinear', align_corners=False)
        loss = criterion(outputs, target_resized)
        loss.backward()
        optimizer.step()
    print(f'\nEpoch {epoch+1}/{num_epochs}, Loss: {loss.item()}', flush=True)

torch.save(model.state_dict(), 'models/frame_interpolation_model.pth')