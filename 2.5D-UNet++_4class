"""
UNet++ 2.5D Segmentation Model for Multiclass Segmentation 🧠

This model performs multiclass segmentation across 4 classes using a nested UNet++ architecture.
It supports deep supervision, 2.5D input (stacked slices), and is built for datasets 
that demand both anatomical fidelity and computational style.

Model inputs are assumed to be in shape [B, C, H, W], where C = number of slices (2.5D).
Model outputs are probability maps over 4 classes (Softmax).

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# This is a simple double Conv2D block with ReLU activation.
# It's the basic building block used throughout UNet++.
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision

        # ---- 2.5D INPUT NOTE ----
        # in_channels should represent the number of slices (e.g. 3, 5, 7), not RGB
        # For example, input shape: [batch_size, in_channels, height, width]

        # ---- OUTPUT NOTE ----
        # out_channels = 4 indicates multiclass segmentation with softmax activation

        # Number of filters for each encoder/decoder level
        filters = [32, 64, 128, 256, 512]

        # ---- ENCODER PATH ----
        # Each block halves spatial resolution and increases feature channels
        self.conv0_0 = ConvBlock(in_channels, filters[0])
        self.conv1_0 = ConvBlock(filters[0], filters[1])
        self.conv2_0 = ConvBlock(filters[1], filters[2])
        self.conv3_0 = ConvBlock(filters[2], filters[3])
        self.conv4_0 = ConvBlock(filters[3], filters[4])

        # Max pooling (2x downsampling)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---- DECODER PATH (Nested skip connections) ----
        # These are your recursive dense pathways
        # Each decoder block combines previous outputs + upsampled lower-level result
        self.conv0_1 = ConvBlock(filters[0] + filters[1], filters[0])
        self.conv1_1 = ConvBlock(filters[1] + filters[2], filters[1])
        self.conv2_1 = ConvBlock(filters[2] + filters[3], filters[2])
        self.conv3_1 = ConvBlock(filters[3] + filters[4], filters[3])

        self.conv0_2 = ConvBlock(filters[0]*2 + filters[1], filters[0])
        self.conv1_2 = ConvBlock(filters[1]*2 + filters[2], filters[1])
        self.conv2_2 = ConvBlock(filters[2]*2 + filters[3], filters[2])

        self.conv0_3 = ConvBlock(filters[0]*3 + filters[1], filters[0])
        self.conv1_3 = ConvBlock(filters[1]*3 + filters[2], filters[1])

        self.conv0_4 = ConvBlock(filters[0]*4 + filters[1], filters[0])

        # ---- FINAL OUTPUT LAYERS ----
        # Deep supervision = auxiliary outputs from intermediate layers
        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        else:
            # Single final output with 4-class softmax for multiclass segmentation
            self.final = nn.Sequential(
                nn.Conv2d(filters[0], out_channels, kernel_size=1),
                nn.Softmax(dim=1)
            )

    # Simple bilinear upsampling block
    def upsample(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # ---- ENCODER ----
        # These are the standard downsampling layers
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # ---- DECODER ----
        # Each x_{i,j} block uses ALL previous x_{i,k} (k < j) and one upsampled x_{i+1,j-1}
        x0_1 = self.conv0_1(torch.cat([x0_0, self.upsample(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.upsample(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.upsample(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.upsample(x4_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.upsample(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.upsample(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.upsample(x3_1)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.upsample(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.upsample(x2_2)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.upsample(x1_3)], dim=1))

        # ---- OUTPUT ----
        # If deep supervision is on, return multiple outputs from different depths
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            # Otherwise return final prediction from the last decoder node
            return self.final(x0_4)
