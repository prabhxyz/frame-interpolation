import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

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

# Load the trained model
loaded_model = FrameInterpolationModel()
loaded_model.load_state_dict(torch.load('models/frame_interpolation_model4.pth'))
loaded_model.eval()

# Inference
with torch.no_grad():
    input_frame1 = Image.open('dataset/test/frame1.jpg')
    input_frame2 = Image.open('dataset/test/frame2.jpg')
    transform = transforms.ToTensor()
    input_frame1 = transform(input_frame1)
    input_frame2 = transform(input_frame2)
    inputs = torch.cat((input_frame1, input_frame2), dim=0).unsqueeze(0)
    output = loaded_model(inputs)
    interpolated_frame = output.squeeze(0).permute(1, 2, 0).numpy()
    interpolated_frame = (interpolated_frame * 255).astype(np.uint8)
    interpolated_frame = Image.fromarray(interpolated_frame)
    output_path = 'output/interpolated_frame.png'
    interpolated_frame.save(output_path)

print(f"Interpolated frame saved at: {output_path}")