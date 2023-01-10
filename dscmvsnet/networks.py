import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.conv import *
import numpy as np

# fixme: Feature Extraction for Guass-Netwon Layer
class ImageConv(nn.Module):
    def __init__(self, base_channels, in_channels=3):
        super(ImageConv, self).__init__()
        self.base_channels = base_channels # 8
        self.out_channels = 8 * base_channels # 64
        self.conv0 = nn.Sequential(
            Conv2d(in_channels, base_channels, 3, 1, padding=1), # 3 → 8   ReLU+BN
            Conv2d(base_channels, base_channels, 3, 1, padding=1), # 8 → 8  ReLU+BN
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2), # 8 → 16 ReLU+BN
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1), # 16 → 16 ReLU+BN
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1), # 16 → 16 ReLU+BN
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2), # 16 → 32 ReLU+BN
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1), # 32 → 32 ReLU+BN
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1, bias=False) # 32 → 32
        )


    def forward(self, imgs):
        out_dict = {}

        conv0 = self.conv0(imgs)
        out_dict["conv0"] = conv0 # save multi-scale features
        conv1 = self.conv1(conv0)
        out_dict["conv1"] = conv1 # save multi-scale features
        conv2 = self.conv2(conv1)
        out_dict["conv2"] = conv2 # save multi-scale features

        return out_dict


# fixme: Informative Feature Extraction Network
class IFENetwork(nn.Module):
    def __init__(self, base_channels, in_channels=3):
        super(IFENetwork, self).__init__()
        self.base_channels = base_channels # 8
        self.out_channels = 8 * base_channels  # 64

        # Downsample
        self.conv0 = ConvBnReLU(in_channels, base_channels, 3, 1, 1)
        self.conv1 = ConvBnReLU(base_channels, base_channels, 3, 1, 1)

        self.conv2 = ConvBnReLU(base_channels, base_channels * 2, 5, 2, 2)
        self.conv3 = ConvBnReLU(base_channels * 2, base_channels * 2, 3, 1, 1)
        self.conv4 = ConvBnReLU(base_channels * 2, base_channels * 2, 3, 1, 1)

        self.conv5 = ConvBnReLU(base_channels * 2, base_channels * 4, 5, 2, 2)
        self.conv6 = ConvBnReLU(base_channels * 4, base_channels * 4, 3, 1, 1)
        self.conv7 = ConvBnReLU(base_channels * 4, base_channels * 4, 3, 1, 1)

        # Upsample
        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        self.conv9 = ConvBnReLU(base_channels * 2, base_channels * 2, 3, 1, 1)
        self.conv10 = ConvBnReLU(base_channels * 2, base_channels * 2, 3, 1, 1)

        self.conv11 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.conv12 = ConvBnReLU(base_channels, base_channels, 3, 1, 1)
        self.conv13 = ConvBnReLU(base_channels, base_channels, 3, 1, 1)

        self.conv14 = ConvBnReLU(base_channels, base_channels * 2, 5, 2, 2)
        self.conv15 = ConvBnReLU(base_channels * 2, base_channels * 2, 3, 1, 1)

        self.conv16 = ConvBnReLU(base_channels * 2, base_channels * 4, 5, 2, 2)
        self.feature = nn.Conv2d(base_channels * 4, base_channels * 4, 3, 1, 1)

    def forward(self, x):
        out_dict = {}

        # down sample
        feature1 = self.conv1(self.conv0(x)) # channel = 8
        feature2 = self.conv4(self.conv3(self.conv2(feature1))) # channel = 16
        x = self.conv7(self.conv6(self.conv5(feature2))) # channel = 32

        # up sample + skip connection
        x = feature2 + self.conv10(self.conv9(self.conv8(x))) # channel = 16
        x = feature1 + self.conv13(self.conv12(self.conv11(x))) # channel = 8
        out_dict['conv0'] = x # channel = 8

        x = self.conv15(self.conv14(x))
        out_dict['conv1'] = x # channel = 16

        x = self.feature(self.conv16(x))
        out_dict['conv2'] = x # channel = 32

        return out_dict


# fixme: Kernel_FTM
class Kernel_FTM(nn.Module):
    def __init__(self, input_channel, kernel_size, factor=4):
        super(Kernel_FTM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 8, kernel_size=7, stride=1, padding=3),  # kernel=7
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1), # kernel=2, stride=2
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),  # kernel=5
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), # kernel=2, stride=2
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_weight = nn.Conv2d(32, kernel_size**2*(factor)**2, 1)
        self.conv_offset = nn.Conv2d(32, 2*kernel_size**2*(factor)**2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        offset = self.conv_offset(x)
        weight = torch.sigmoid(self.conv_weight(x))

        return weight, offset


# fixme: resample_data
def resample_data(input, s):
    """
        input: torch.floatTensor (N, C, H, W)
        s: int (resample factor)
    """

    assert (not input.size(2) % s and not input.size(3) % s)

    if input.size(1) == 3:
        # bgr2gray (same as opencv conversion matrix)
        input = (0.299 * input[:, 2] + 0.587 * input[:, 1] + 0.114 * input[:, 0]).unsqueeze(1)

    out = torch.cat([input[:, :, i::s, j::s] for i in range(s) for j in range(s)], dim=1)

    """
        out: torch.floatTensor (N, s**2, H/s, W/s)
    """
    return out

def grid_generator(k, r, n):
    """
    	grid_generator
	    Parameters
	    ---------
	    f : filter_size, int
	    k: kernel_size, int
	    n: number of grid, int
	    Returns
	    -------
	    torch.Tensor. shape = (n, 2, k, k)
    """
    grid_x, grid_y = torch.meshgrid([torch.linspace(k//2, k//2+r-1, steps=r),
                                     torch.linspace(k//2, k//2+r-1, steps=r)])
    grid = torch.stack([grid_x,grid_y],2).view(r,r,2)

    return grid.unsqueeze(0).repeat(n,1,1,1).cuda()

# fixme: Sparse Depth Map → Dense Depth Map  使用FDKN Module  Feature Transfer Module
class FTM(nn.Module):
    def __init__(self, kernel_size, filter_size, residual=True):
        super(FTM, self).__init__()
        self.factor = 4
        self.ImageKernel = Kernel_FTM(input_channel=16, kernel_size=kernel_size, factor=self.factor)
        self.DepthKernel = Kernel_FTM(input_channel=1, kernel_size=kernel_size, factor=self.factor)
        self.residual = residual
        self.kernel_size = kernel_size
        self.filter_size = filter_size

    def forward(self, depth, image):
        """
        :param depth:   [B, 1, 128, 160]
        :param image:   [B, C, 512, 640]
        :return:
        """
        re_img = resample_data(image, self.factor) # [B, factor**2, H/factor, W/factor]  [B, 16, 128, 160]

        img_weight, img_offset = self.ImageKernel(re_img) # weight:[B, 144, 32, 40]  offset:[B, 288, 32, 40]
        dp_weight, dp_offset = self.DepthKernel(depth) # weight:[B, 144, 32, 40]  offset:[B, 288, 32, 40]

        weight = img_weight * dp_weight  # [B, 144, 32, 40]
        offset = img_offset * dp_offset  # [B, 288, 32, 40]

        ps = nn.PixelShuffle(4)
        weight = ps(weight) # [B, 9, 128, 160]
        offset = ps(offset) # [B, 18, 128, 160]

        if self.residual:
            weight -= torch.mean(weight, 1).unsqueeze(1).expand_as(weight)
        else:
            weight /= torch.sum(weight, 1).unsqueeze(1).expand_as(weight)

        b, h, w = depth.size(0), depth.size(2), depth.size(3) # 1, 128, 160
        k = self.filter_size # 15
        r = self.kernel_size # 3
        hw = h * w # 20480

        # weighted average
        # (b, 2*r**2, h, w) -> (b*hw, r, r, 2)
        offset = offset.permute(0, 2, 3, 1).contiguous().view(b * hw, r, r, 2)  # [20480, 3, 3, 2]
        # (b, r**2, h, w) -> (b*hw, r**2, 1)
        weight = weight.permute(0, 2, 3, 1).contiguous().view(b * hw, r * r, 1) # [20480, 9, 1]

        # (b*hw, r, r, 2)
        grid = grid_generator(k, r, b * hw) # [20480, 3, 3, 2]
        coord = grid + offset # [20480, 3, 3, 2]
        coord = (coord / k * 2) - 1 # [20480, 3, 3, 2]

        # (b, k**2, hw) -> (b*hw, 1, k, k)
        depth_col = F.unfold(depth, k, padding=k // 2).permute(0, 2, 1).contiguous().view(b * hw, 1, k, k)  # [20480, 1, 15, 15]

        # (b*hw, 1, k, k), (b*hw, r, r, 2) => (b*hw, 1, r^2)
        depth_sampled = F.grid_sample(depth_col, coord).view(b * hw, 1, -1) # [20480, 1, 9]

        # (b*w*h, 1, r^2) x (b*w*h, r^2, 1) => (b, 1, h, w)
        out = torch.bmm(depth_sampled, weight).view(b, 1, h, w) # [1, 1, 128, 160]

        if self.residual:
            out += depth

        return out


# fixme: Flatten for Channel Attention
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# fixme: Channel Attention
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        # MLP
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        channel_att_raw = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d(x, kernel_size=(x.size(2), x.size(3), x.size(4)),stride=(x.size(2), x.size(3), x.size(4))) # [B, C, 1, 1, 1]
                channel_att_raw = self.mlp(avg_pool) # [B, C]
            elif pool_type == 'max':
                max_pool = F.max_pool3d(x, kernel_size=(x.size(2), x.size(3), x.size(4)),stride=(x.size(2), x.size(3), x.size(4))) # [B, C, 1, 1, 1]
                channel_att_raw = self.mlp(max_pool) # [B, C]

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = F.sigmoid(channel_att_sum) # [B, C]
        scale = scale.unsqueeze(2).unsqueeze(3).unsqueeze(3).expand_as(x) # [B, C, X.D, X.H, X.W]
        return x * scale

# fixme: Feature Extraction for Spatial Attention
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# fixme: Channel Max Pooling
class ChannelPool(nn.Module):
    def __init__(self):
        super(ChannelPool, self).__init__()

    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

# fixme: Spatial Depth Attention
class SpatialDepthGate(nn.Module):
    def __init__(self):
        super(SpatialDepthGate, self).__init__()
        kernel_size = 7
        self.channel_pool = ChannelPool()
        self.channel_conv = BasicConv(2, 1, kernel_size=(1, kernel_size, kernel_size), stride=1, padding=(0, (kernel_size-1) // 2, (kernel_size-1) // 2), relu=False)  # 后面为了减少参数量可以考虑将它换为可分离卷积
        self.depth_conv = BasicConv(1, 1, kernel_size=(kernel_size, 1, 1), stride=1, padding=((kernel_size-1) // 2, 0, 0), relu=False)
        self.overall_conv = BasicConv(1, 1, kernel_size=(kernel_size,kernel_size,kernel_size), stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x): # x [B, C. X.D, X.H, X.W]
        compress = self.channel_pool(x) # [B, 2, X.D, X.H, X.W]  [B, 32, 12, 16,20]
        compress = self.channel_conv(compress) # [B, 1, 12 ,16, 20]
        compress = self.depth_conv(compress) # [B, 1, 12 ,16, 20]
        compress = self.overall_conv(compress) # [1, 1, 12, 16, 20]
        scale = F.sigmoid(compress) # [1, 1, 12, 16, 20]
        return x * scale

# fixme: 3D Attention Module
class 3DAModule(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_type=['avg', 'max'], no_spatial_depth=False):
        super(3DAModule, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_type)
        self.no_spatial_depth = no_spatial_depth
        if not no_spatial_depth:
            self.SpatialDepthGate = SpatialDepthGate()

    def forward(self, x):
        x = self.ChannelGate(x)
        if not self.no_spatial_depth:
            x = self.SpatialDepthGate(x)
        return x


# fixme: use depth separable convolution cost volume regularization skip connection 3D U-Net
class DSCVolumeReg(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(DSCVolumeReg, self).__init__()
        self.in_channels = in_channels # 32
        self.base_channels = base_channels # 8
        self.output_channels = base_channels * 8 # 64

        # Basic Conv
        self.conv0_1 = DSConv3d(in_channels, base_channels, stride=1) # 32 → 8

        self.conv1_1 = DSConv3d(base_channels * 2, base_channels * 2, stride=1) # 16 → 16
        self.conv2_1 = DSConv3d(base_channels * 4, base_channels * 4, stride=1) # 16 → 16
        self.conv3_1 = DSConv3d(base_channels * 8, base_channels * 8, stride=1) # 16 → 16


        # Downsample
        self.conv1_0 = DSConv3d(in_channels, base_channels * 2, stride=2) # 32 → 16
        self.conv2_0 = DSConv3d(base_channels * 2, base_channels * 4, stride=2) # 16 → 32
        self.conv3_0 = DSConv3d(base_channels * 4, base_channels * 8, stride=2) # 32 → 64

        # Upsample
        self.conv4_0 = DSDeConv3d(base_channels * 8, base_channels * 4, stride=2) # 64 → 32
        self.conv5_0 = DSDeConv3d(base_channels * 4, base_channels * 2, stride=2) # 32 → 16
        self.conv6_0 = DSDeConv3d(base_channels * 2, base_channels, stride=2) # 16 → 8

        self.conv6_2 = nn.Conv3d(base_channels, 1, 3, padding=1, bias=False)  # 8 → 1

        # 3DAModule
        self.attention_block_1 = 3DAModule(self.base_channels * 4, reduction_ratio=8, pool_type=['avg', 'max'], no_spatial_depth=False)
        self.attention_block_2 = 3DAModule(self.base_channels * 2, reduction_ratio=8, pool_type=['avg', 'max'], no_spatial_depth=False)
        self.attention_block_3 = 3DAModule(self.base_channels, reduction_ratio=8, pool_type=['avg', 'max'], no_spatial_depth=False)


    def forward(self, x):
        conv0_1 = self.conv0_1(x)  # 32 → 8 [B, 32(C), 48(D), 64(H/2), 80(W/2)] → [B, 8(C), 48, 64, 80]

        conv1_0 = self.conv1_0(x)  # 32 → 16  [B, 8(C), 48(D), 64(H/2), 80(W/2)] → [B, 16(C), 24, 32, 40]
        conv2_0 = self.conv2_0(conv1_0)  # 16 → 32  [B, 16(C), 24, 32, 40] → [B, 32, 12, 16, 20]
        conv3_0 = self.conv3_0(conv2_0)  # 32 → 64  [B, 32(C), 12, 16, 20] → [B, 64, 6, 8, 10]

        conv1_1 = self.conv1_1(conv1_0)  # 16 → 16  [B, 16(C), 24, 32, 40] → [B, 16, 24, 32, 40]
        conv2_1 = self.conv2_1(conv2_0)  # 32 → 32  [B, 32, 12, 16, 20] → [B, 32, 12, 16, 20]
        conv3_1 = self.conv3_1(conv3_0)  # 64 → 64  [B, 64, 6, 8, 10] → [B, 64, 6, 8, 10]

        conv4_0 = self.conv4_0(conv3_1)  # 64 → 32  [B, 64, 6, 8, 10] → [B, 32, 12, 16, 20]

        # skip connection
        conv5_0 = self.conv5_0(self.attention_block_1(conv4_0 + conv2_1))  # 32 → 16  [B, 32, 12, 16, 20] → [B, 16, 24, 32, 40]
        conv6_0 = self.conv6_0(self.attention_block_2(conv5_0 + conv1_1))  # 16 → 8  [B, 16, 24, 32, 40] → [B, 8, 48, 64, 80]

        conv6_2 = self.conv6_2(self.attention_block_3(conv6_0 + conv0_1))  # 8 → 1 probability volume  [B, 8, 48 ,64, 80] → [B, 1, 48, 64, 80]

        return conv6_2



class MAELoss(nn.Module):
    def forward(self, pred_depth_image, gt_depth_image, depth_interval):
        """non zero mean absolute loss for one batch"""
        # shape = list(pred_depth_image)
        depth_interval = depth_interval.view(-1)
        mask_valid = (~torch.eq(gt_depth_image, 0.0)).type(torch.float)
        denom = torch.sum(mask_valid, dim=(1, 2, 3)) + 1e-7
        masked_abs_error = mask_valid * torch.abs(pred_depth_image - gt_depth_image)
        masked_mae = torch.sum(masked_abs_error, dim=(1, 2, 3))
        masked_mae = torch.sum((masked_mae / depth_interval) / denom)

        return masked_mae


class Valid_MAELoss(nn.Module):
    def __init__(self, valid_threshold=2.0):
        super(Valid_MAELoss, self).__init__()
        self.valid_threshold = valid_threshold

    def forward(self, pred_depth_image, gt_depth_image, depth_interval, before_depth_image):
        """non zero mean absolute loss for one batch"""
        # shape = list(pred_depth_image)
        pred_height = pred_depth_image.size(2)
        pred_width = pred_depth_image.size(3)
        depth_interval = depth_interval.view(-1)
        mask_true = (~torch.eq(gt_depth_image, 0.0)).type(torch.float)
        before_hight = before_depth_image.size(2)
        if before_hight != pred_height:
            before_depth_image = F.interpolate(before_depth_image, (pred_height, pred_width))
        diff = torch.abs(gt_depth_image - before_depth_image) / depth_interval.view(-1, 1, 1, 1)
        mask_valid = (diff < self.valid_threshold).type(torch.float)
        mask_valid = mask_true * mask_valid
        denom = torch.sum(mask_valid, dim=(1, 2, 3)) + 1e-7
        masked_abs_error = mask_valid * torch.abs(pred_depth_image - gt_depth_image)
        masked_mae = torch.sum(masked_abs_error, dim=(1, 2, 3))
        masked_mae = torch.sum((masked_mae / depth_interval) / denom)

        return masked_mae
