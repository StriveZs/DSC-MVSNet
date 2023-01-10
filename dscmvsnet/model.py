import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import *
from functions.functions import get_pixel_grids, get_propability_map
from utils.feature_fetcher import FeatureFetcher, FeatureGradFetcher, PointGrad, ProjectUVFetcher


class DSCMVSNet(nn.Module):
    def __init__(self,
                 img_base_channels=8,
                 vol_base_channels=8,
                 flow_channels=(64, 64, 16, 1),
                 k=16,
                 ):
        super(DSCMVSNet, self).__init__()
        self.k = k

        self.feature_fetcher = FeatureFetcher()
        self.feature_grad_fetcher = FeatureGradFetcher()
        self.point_grad_fetcher = PointGrad()

        self.coarse_img_conv = IFENetwork(img_base_channels) # Informative Feautre Extraction Network
        self.coarse_vol_conv = DSCVolumeReg(img_base_channels * 4, vol_base_channels) # DSC-Attention 3D UNet
        self.feature_transfer_module = FTM(3, 15, True)
        self.flow_img_conv = ImageConv(img_base_channels)  # Feature Extraction for Gauss-Netwon Layer

    def forward(self, data_batch, img_scales, inter_scales, isGN, isTest=False):
        """
        model forward
        data_batch:
            img_list: [batch, 3, 3, 512, 640]
            cam_params_list: [batch, View, 2, 4, 5]
            gt_depth_img: [batch, 1, 128, 160] reference images ground truth depth map
            depth_list: [batch, 3, 1, 128, 160]
            mean: [batch, 3]
            std:  [batch, 3]
        :param data_batch: img_list、cam_params_list、gt_depth_img、depth_list、mean、std
        :param img_scales: (0.25, 0.5)
        :param inter_scales: (0.75, 0.375)
        :param isGN: use or not use Gauss-Netwon Layer
        :param isTest: test or not test
        :return:
        """
        preds = collections.OrderedDict()
        img_list = data_batch["img_list"] # ref+source   [B, V, C, 960, 1280]
        cam_params_list = data_batch["cam_params_list"] # camera parameters

        cam_extrinsic = cam_params_list[:, :, 0, :3, :4].clone()  # [B, V, 3, 4] camera external parameters
        R = cam_extrinsic[:, :, :3, :3] # rotation matrix [B, V, 3, 3]
        t = cam_extrinsic[:, :, :3, 3].unsqueeze(-1) # translation matrix [B, V, 3, 1]
        R_inv = torch.inverse(R)
        cam_intrinsic = cam_params_list[:, :, 1, :3, :3].clone() # [B, V, 3, 3] camera internal parameters

        if isTest:
            cam_intrinsic[:, :, :2, :3] = cam_intrinsic[:, :, :2, :3] / 4.0

        # depth number and depth interval
        depth_start = cam_params_list[:, 0, 1, 3, 0] # [425, 425, 425, 425]
        depth_interval = cam_params_list[:, 0, 1, 3, 1] # [10.6, 10.6, 10.6, 10.6]
        num_depth = cam_params_list[0, 0, 1, 3, 2].long() # depth number
        depth_end = depth_start + (num_depth - 1) * depth_interval # [923.2, 923.2, 923.2, 923.2]

        batch_size, num_view, img_channel, img_height, img_width = list(img_list.size()) # bz=4 view=3 channel=3 height=512 weight=640

        # Informative feature extraction network to obtain coarse feature maps
        coarse_feature_maps = []
        for i in range(num_view):
            curr_img = img_list[:, i, :, :, :]
            curr_feature_map = self.coarse_img_conv(curr_img)["conv2"]  # [b, 32, 128, 160]
            coarse_feature_maps.append(curr_feature_map)

        # feature maps stack, [b,c,w,h] → [b,m,c,w,h]
        feature_list = torch.stack(coarse_feature_maps, dim=1) # [batch, view, 32, 128, 160]
        feature_channels, feature_height, feature_width = list(curr_feature_map.size())[1:]

        # depth hypothesis planes
        depths = [] # [1， 1，depth number, 1]
        for i in range(batch_size):
            depths.append(torch.linspace(depth_start[i], depth_end[i], num_depth, device=img_list.device) \
                          .view(1, 1, num_depth, 1))
        depths = torch.stack(depths, dim=0)  # [B, 1, 1, D, 1]

        feature_map_indices_grid = get_pixel_grids(feature_height, feature_width) # [view, height × weight]
        feature_map_indices_grid = feature_map_indices_grid.view(1, 3, feature_height, feature_width)[:, :, ::2, ::2].contiguous() # [1, V, H/2, W/2] [1, 3, 64, 80]
        feature_map_indices_grid = feature_map_indices_grid.view(1, 1, 3, -1).expand(batch_size, 1, 3, -1).to(img_list.device) # [B, 1, V, H/2 × W/2] [B, 1, 3, 5120]

        ref_cam_intrinsic = cam_intrinsic[:, 0, :, :].clone() # [4, 3, 3]
        uv = torch.matmul(torch.inverse(ref_cam_intrinsic).unsqueeze(1), feature_map_indices_grid)  # (B, 1, 3, FH*FW) [batch, 1, 3, 5210]
        # cam_points 2457600 = 19200 * 128   240 * 320 / 4 = 19200
        cam_points = (uv.unsqueeze(3) * depths).view(batch_size, 1, 3, -1)  # (B, 1, 3, D*FH*FW) [B, 1, V, H/2 × W/2]
        world_points = torch.matmul(R_inv[:, 0:1, :, :], cam_points - t[:, 0:1, :, :]).transpose(1, 2).contiguous() \
            .view(batch_size, 3, -1)  # (B, 3, D*FH*FW)

        preds["world_points"] = world_points # [B, V, 245760]  128*160*48 / 4

        num_world_points = world_points.size(-1) # construct sparse cost volume
        assert num_world_points == feature_height * feature_width * num_depth / 4

        # feature list [1, 7, 32, 240, 320] world_points []
        point_features = self.feature_fetcher(feature_list, world_points, cam_intrinsic, cam_extrinsic) # [B, V, 32, 2457650]
        ref_feature = coarse_feature_maps[0] # [B, 32, 128, 160]

        ref_feature = ref_feature[:, :, ::2,::2].contiguous() # [B, 32, 64, 80]

        ref_feature = ref_feature.unsqueeze(2).expand(-1, -1, num_depth, -1, -1)\
                        .contiguous().view(batch_size,feature_channels,-1) # [B, 32, (128/2)×(160/2)×48=245760]
        point_features[:, 0, :, :] = ref_feature  # [B, V, 32, 245760]

        avg_point_features = torch.mean(point_features, dim=1) # [B, 32, 245760]
        avg_point_features_2 = torch.mean(point_features ** 2, dim=1) # [B, 32, 245760]

        point_features = avg_point_features_2 - (avg_point_features ** 2)

        cost_volume = point_features.view(batch_size, feature_channels, num_depth, feature_height // 2, feature_width // 2) # [B, 32(C), 48(D), 60(H/2), 80(W/2)]手动计算sparse cost volume

        # Cost Volume Regularization
        filtered_cost_volume = self.coarse_vol_conv(cost_volume).squeeze(1) # [B, 48, 64, 80] sparse cost volume regularization

        probability_volume = F.softmax(-filtered_cost_volume, dim=1) # [B, 48, 64, 80] probability volume regression
        depth_volume = []
        for i in range(batch_size):
            depth_array = torch.linspace(depth_start[i], depth_end[i], num_depth, device=depth_start.device)
            depth_volume.append(depth_array)
        depth_volume = torch.stack(depth_volume, dim=0)  # [B, D]
        depth_volume = depth_volume.view(batch_size, num_depth, 1, 1).expand(probability_volume.shape) # [B, 48, 64, 80]
        pred_depth_img = torch.sum(depth_volume * probability_volume, dim=1).unsqueeze(1)  # (B, 1, FH, FW) [B, 1, 64, 80]

        prob_map = get_propability_map(probability_volume, pred_depth_img, depth_start, depth_interval) # [B, 1, 64, 80]

        # Feature Transfer Module
        pred_depth_img = F.interpolate(pred_depth_img, (feature_height, feature_width),
                                       mode="bicubic")
        prob_map = F.interpolate(prob_map, (feature_height, feature_width),
                                 mode="bicubic")
        pred_depth_img = self.feature_transfer_module(pred_depth_img, img_list[:, 0, :, :,
                                                              :])  # [B, 1, 128, 160]  将通过最近邻插值[B, 1, 128, 160]的粗糙深度图 由结合图像特征 来生成refined之后的[B, 1, 128, 160]的深度图

        preds["coarse_depth_map"] = pred_depth_img
        preds["coarse_prob_map"] = prob_map

        # Guass-Netwon Layer
        if isGN:
            feature_pyramids = {}
            chosen_conv = ["conv1", "conv2"]
            for conv in chosen_conv:
                feature_pyramids[conv] = []
            for i in range(num_view):
                curr_img = img_list[:, i, :, :, :]
                curr_feature_pyramid = self.flow_img_conv(curr_img)
                for conv in chosen_conv:
                    feature_pyramids[conv].append(curr_feature_pyramid[conv])
            for conv in chosen_conv:
                feature_pyramids[conv] = torch.stack(feature_pyramids[conv], dim=1)  # conv1: [B, 3, 16, 256, 320]  conv2: [B, 3, 32, 128, 160]

            if isTest:
                for conv in chosen_conv:
                    feature_pyramids[conv] = torch.detach(feature_pyramids[conv])


            def gn_update(estimated_depth_map, interval, image_scale, it):
                nonlocal chosen_conv
                flow_height, flow_width = list(estimated_depth_map.size())[2:] # flow_height = 128 flow_width = 160
                if flow_height != int(img_height * image_scale):
                    flow_height = int(img_height * image_scale)
                    flow_width = int(img_width * image_scale)
                    estimated_depth_map = F.interpolate(estimated_depth_map, (flow_height, flow_width), mode="nearest")
                else:
                    # if it is the same size return directly
                    return estimated_depth_map
                    # pass
                
                if isTest:
                    estimated_depth_map = estimated_depth_map.detach()

                cam_intrinsic = cam_params_list[:, :, 1, :3, :3].clone() # [B, V, 3, 3]
                if isTest:
                    cam_intrinsic[:, :, :2, :3] *= image_scale
                else:
                    cam_intrinsic[:, :, :2, :3] *= (4 * image_scale)

                ref_cam_intrinsic = cam_intrinsic[:, 0, :, :].clone() # [B, 3, 3]
                feature_map_indices_grid = get_pixel_grids(flow_height, flow_width) \
                    .view(1, 1, 3, -1).expand(batch_size, 1, 3, -1).to(img_list.device)  # [B, 1, V, flow_height × flow_width = 81920]

                uv = torch.matmul(torch.inverse(ref_cam_intrinsic).unsqueeze(1),
                                  feature_map_indices_grid)  # (B, 1, 3, FH*FW) # [B, 1, V, flow_height × flow_width = 81920]

                interval_depth_map = estimated_depth_map # [B, 1, 256, 320]
                cam_points = (uv * interval_depth_map.view(batch_size, 1, 1, -1)) # [B, 1, V, flow_height × flow_width = 81920]
                world_points = torch.matmul(R_inv[:, 0:1, :, :], cam_points - t[:, 0:1, :, :]).transpose(1, 2) \
                    .contiguous().view(batch_size, 3, -1)  # (B, 3, D*FH*FW)  [B, V, flow_height × flow_width = 81920]

                grad_pts = self.point_grad_fetcher(world_points, cam_intrinsic, cam_extrinsic) # [B, V, flow_height × flow_width = 81920, 2, 3]

                R_tar_ref = torch.bmm(R.view(batch_size * num_view, 3, 3),
                                      R_inv[:, 0:1, :, :].repeat(1, num_view, 1, 1).view(batch_size * num_view, 3, 3)) # [12, 3, 3]

                R_tar_ref = R_tar_ref.view(batch_size, num_view, 3, 3) # [4, 3, 3, 3]
                d_pts_d_d = uv.unsqueeze(-1).permute(0, 1, 3, 2, 4).contiguous().repeat(1, num_view, 1, 1, 1)
                d_pts_d_d = R_tar_ref.unsqueeze(2) @ d_pts_d_d
                d_uv_d_d = torch.bmm(grad_pts.view(-1, 2, 3), d_pts_d_d.view(-1, 3, 1)).view(batch_size, num_view, 1,
                                                                                             -1, 2, 1)
                all_features = []
                for conv in chosen_conv:
                    curr_feature = feature_pyramids[conv]
                    c, h, w = list(curr_feature.size())[2:] # c:32 h:128 w:160
                    curr_feature = curr_feature.contiguous().view(-1, c, h, w)
                    curr_feature = F.interpolate(curr_feature, (flow_height, flow_width), mode="bilinear")
                    curr_feature = curr_feature.contiguous().view(batch_size, num_view, c, flow_height, flow_width)

                    all_features.append(curr_feature)

                all_features = torch.cat(all_features, dim=2) # [B, V, C×V, 256, 320] [B, 3, 48, 256, 320]

                if isTest:
                    point_features, point_features_grad = \
                        self.feature_grad_fetcher.test_forward(all_features, world_points, cam_intrinsic, cam_extrinsic)
                else:
                    point_features, point_features_grad = \
                        self.feature_grad_fetcher(all_features, world_points, cam_intrinsic, cam_extrinsic)

                c = all_features.size(2)
                d_uv_d_d_tmp = d_uv_d_d.repeat(1, 1, c, 1, 1, 1)
                # print("d_uv_d_d tmp size:", d_uv_d_d_tmp.size())
                J = point_features_grad.view(-1, 1, 2) @ d_uv_d_d_tmp.view(-1, 2, 1)
                J = J.view(batch_size, num_view, c, -1, 1)[:, 1:, ...].contiguous()\
                    .permute(0, 3, 1, 2, 4).contiguous().view(-1, c * (num_view - 1), 1)

                # print(J.size())
                resid = point_features[:, 1:, ...] - point_features[:, 0:1, ...]
                first_resid = torch.sum(torch.abs(resid), dim=(1, 2))
                # print(resid.size())
                resid = resid.permute(0, 3, 1, 2).contiguous().view(-1, c * (num_view - 1), 1)

                J_t = torch.transpose(J, 1, 2)
                H = J_t @ J
                b = -J_t @ resid
                delta = b / (H + 1e-6)
                # #print(delta.size())
                _, _, h, w = estimated_depth_map.size()
                flow_result = estimated_depth_map  + delta.view(-1, 1, h, w)

                # check update results
                interval_depth_map = flow_result
                cam_points = (uv * interval_depth_map.view(batch_size, 1, 1, -1))
                world_points = torch.matmul(R_inv[:, 0:1, :, :], cam_points - t[:, 0:1, :, :]).transpose(1, 2) \
                    .contiguous().view(batch_size, 3, -1)  # (B, 3, D*FH*FW)

                point_features = \
                    self.feature_fetcher(all_features, world_points, cam_intrinsic, cam_extrinsic)

                resid = point_features[:, 1:, ...] - point_features[:, 0:1, ...]
                second_resid = torch.sum(torch.abs(resid), dim=(1, 2))
                # print(first_resid.size(), second_resid.size())

                # only accept good update
                flow_result = torch.where((second_resid < first_resid).view(batch_size, 1, flow_height, flow_width),
                                          flow_result, estimated_depth_map)
                return flow_result

            for i, (img_scale, inter_scale) in enumerate(zip(img_scales, inter_scales)):
                if isTest:
                    pred_depth_img = torch.detach(pred_depth_img)
                    print("update: {}".format(i))
                flow = gn_update(pred_depth_img, inter_scale* depth_interval, img_scale, i) # [B, 1, 128, 160] flow 为 flow refine之后的depth map
                preds["flow{}".format(i+1)] = flow
                pred_depth_img = flow

        return preds

# fixme: loss definition
class DSCMVSNetLoss(nn.Module):
    def __init__(self, valid_threshold):
        super(DSCMVSNetLoss, self).__init__()
        self.maeloss = MAELoss()
        self.valid_maeloss = Valid_MAELoss(valid_threshold)

    def forward(self, preds, labels, isFlow):
        gt_depth_img = labels["gt_depth_img"]
        depth_interval = labels["cam_params_list"][:, 0, 1, 3, 1]

        coarse_depth_map = preds["coarse_depth_map"]
        resize_gt_depth = F.interpolate(gt_depth_img, (coarse_depth_map.shape[2], coarse_depth_map.shape[3]))
        coarse_loss = self.maeloss(coarse_depth_map, resize_gt_depth, depth_interval)

        losses = {}
        losses["coarse_loss"] = coarse_loss

        if isFlow:
            flow1 = preds["flow1"]
            resize_gt_depth = F.interpolate(gt_depth_img, (flow1.shape[2], flow1.shape[3]))
            flow1_loss = self.maeloss(flow1, resize_gt_depth, 0.75 * depth_interval)
            losses["flow1_loss"] = flow1_loss

            flow2 = preds["flow2"]
            resize_gt_depth = F.interpolate(gt_depth_img, (flow2.shape[2], flow2.shape[3]))
            flow2_loss = self.maeloss(flow2, resize_gt_depth, 0.375 * depth_interval)
            losses["flow2_loss"] = flow2_loss

        for k in losses.keys():
            losses[k] /= float(len(losses.keys()))

        return losses


def cal_less_percentage(pred_depth, gt_depth, depth_interval, threshold):
    shape = list(pred_depth.size())
    mask_valid = (~torch.eq(gt_depth, 0.0)).type(torch.float)
    denom = torch.sum(mask_valid) + 1e-7
    interval_image = depth_interval.view(-1, 1, 1, 1).expand(shape)
    abs_diff_image = torch.abs(pred_depth - gt_depth) / interval_image

    pct = mask_valid * (abs_diff_image <= threshold).type(torch.float)

    pct = torch.sum(pct) / denom

    return pct


def cal_valid_less_percentage(pred_depth, gt_depth, before_depth, depth_interval, threshold, valid_threshold):
    shape = list(pred_depth.size())
    mask_true = (~torch.eq(gt_depth, 0.0)).type(torch.float)
    interval_image = depth_interval.view(-1, 1, 1, 1).expand(shape)
    abs_diff_image = torch.abs(pred_depth - gt_depth) / interval_image

    if before_depth.size(2) != shape[2]:
        before_depth = F.interpolate(before_depth, (shape[2], shape[3]))

    diff = torch.abs(before_depth - gt_depth) / interval_image
    mask_valid = (diff < valid_threshold).type(torch.float)
    mask_valid = mask_valid * mask_true

    denom = torch.sum(mask_valid) + 1e-7
    pct = mask_valid * (abs_diff_image <= threshold).type(torch.float)

    pct = torch.sum(pct) / denom

    return pct


# fixme: evaluation metric
class DSCMVSNetMetric(nn.Module):
    def __init__(self, valid_threshold):
        super(DSCMVSNetMetric, self).__init__()
        self.valid_threshold = valid_threshold

    def forward(self, preds, labels, isFlow):
        gt_depth_img = labels["gt_depth_img"]
        depth_interval = labels["cam_params_list"][:, 0, 1, 3, 1]

        coarse_depth_map = preds["coarse_depth_map"]
        resize_gt_depth = F.interpolate(gt_depth_img, (coarse_depth_map.shape[2], coarse_depth_map.shape[3]))

        less_one_pct_coarse = cal_less_percentage(coarse_depth_map, resize_gt_depth, depth_interval, 1.0)
        less_three_pct_coarse = cal_less_percentage(coarse_depth_map, resize_gt_depth, depth_interval, 3.0)

        metrics = {
            "<1_pct_cor": less_one_pct_coarse,
            "<3_pct_cor": less_three_pct_coarse,
        }

        if isFlow:
            flow1 = preds["flow1"]
            resize_gt_depth = F.interpolate(gt_depth_img, (flow1.shape[2], flow1.shape[3]))

            less_one_pct_flow1 = cal_valid_less_percentage(flow1, resize_gt_depth, coarse_depth_map,
                                                           0.75 * depth_interval, 1.0, self.valid_threshold)
            less_three_pct_flow1 = cal_valid_less_percentage(flow1, resize_gt_depth, coarse_depth_map,
                                                             0.75 * depth_interval, 3.0, self.valid_threshold)

            metrics["<1_pct_flow1"] = less_one_pct_flow1
            metrics["<3_pct_flow1"] = less_three_pct_flow1

            flow2 = preds["flow2"]
            resize_gt_depth = F.interpolate(gt_depth_img, (flow2.shape[2], flow2.shape[3]))

            less_one_pct_flow2 = cal_valid_less_percentage(flow2, resize_gt_depth, flow1,
                                                           0.375 * depth_interval, 1.0, self.valid_threshold)
            less_three_pct_flow2 = cal_valid_less_percentage(flow2, resize_gt_depth, flow1,
                                                             0.375 * depth_interval, 3.0, self.valid_threshold)

            metrics["<1_pct_flow2"] = less_one_pct_flow2
            metrics["<3_pct_flow2"] = less_three_pct_flow2

        return metrics


# fixme: build DSC-MVSNet
def build_dscmvsnet(cfg):
    net = DSCMVSNet(
        img_base_channels=cfg.MODEL.IMG_BASE_CHANNELS, # 8
        vol_base_channels=cfg.MODEL.VOL_BASE_CHANNELS, # 8
        flow_channels=cfg.MODEL.FLOW_CHANNELS, # (64, 64, 16, 1)
    )

    loss_fn = DSCMVSNetLoss(
        valid_threshold=cfg.MODEL.VALID_THRESHOLD,
    )

    metric_fn = DSCMVSNetMetric(
        valid_threshold=cfg.MODEL.VALID_THRESHOLD,
    )

    return net, loss_fn, metric_fn


