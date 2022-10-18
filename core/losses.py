import torch
import torch.nn as nn
import numpy as np
import visibility
from depth_completion import sparse_to_dense

def sequence_loss(flow_preds, flow_gt, gamma=0.8, MAX_FLOW=400):
    """ Loss function defined over sequence of flow predictions """

    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    Mask = torch.zeros([flow_gt.shape[0], flow_gt.shape[1], flow_gt.shape[2],
                        flow_gt.shape[3]]).to(flow_gt.device)
    mask = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)
    valid = mask & (mag < MAX_FLOW)
    Mask[:, 0, :, :] = valid
    Mask[:, 1, :, :] = valid
    Mask = Mask != 0
    mask_sum = torch.sum(mask, dim=[1, 2])

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        Loss_reg = (flow_preds[i] - flow_gt) * Mask
        Loss_reg = torch.norm(Loss_reg, dim=1)
        Loss_reg = torch.sum(Loss_reg, dim=[1, 2])
        Loss_reg = Loss_reg / mask_sum
        flow_loss += i_weight * Loss_reg.mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def normal_loss(pred_flows, gt_flows, cam_mats, lidar_input):
    device = gt_flows.device

    loss = 0.

    N = lidar_input.shape[0]
    flow_up = pred_flows[-1]
    for i in range(N):
        depth_img = lidar_input[i].unsqueeze(0)
        flow = flow_up[i].unsqueeze(0)
        gt_flow = gt_flows[i].unsqueeze(0)
        cam_params = cam_mats[i, :]
        fx, fy = cam_params[0], cam_params[1]

        output = torch.zeros(flow.shape).to(device)
        pred_depth_img = torch.zeros(depth_img.shape).to(device)
        pred_depth_img += 1000.
        output = visibility.image_warp_index(depth_img.to(device), flow.int().to(device), pred_depth_img,
                                             output,
                                             depth_img.shape[3], depth_img.shape[2])
        pred_depth_img[pred_depth_img == 1000.] = 0.

        output2 = torch.zeros(flow.shape).to(device)
        gt_depth_img = torch.zeros(depth_img.shape).to(device)
        gt_depth_img += 1000.
        output2 = visibility.image_warp_index(depth_img.to(device), gt_flow.int().to(device), gt_depth_img,
                                              output2,
                                              depth_img.shape[3], depth_img.shape[2])
        gt_depth_img[gt_depth_img == 1000.] = 0.

        gt_depth_img_dilate = sparse_to_dense(gt_depth_img[0, 0, :, :].cpu().numpy().astype(np.float32))
        gt_depth_img_dilate = torch.from_numpy(gt_depth_img_dilate).to(device).unsqueeze(0)
        pred_depth_img_dilate = sparse_to_dense(pred_depth_img[0, 0, :, :].cpu().numpy().astype(np.float32))
        pred_depth_img_dilate = torch.from_numpy(pred_depth_img_dilate).to(device).unsqueeze(0)

        ## choose common points
        mask1_1 = output2[0, 0, :, :] > 0
        mask1_2 = output2[0, 1, :, :] > 0
        mask1 = mask1_1 + mask1_2
        mask2_1 = output[0, 0, :, :] > 0
        mask2_2 = output[0, 1, :, :] > 0
        mask2 = mask2_1 + mask2_2
        mask = mask1 * mask2
        Mask = torch.cat((mask.unsqueeze(0), mask.unsqueeze(0)), dim=0).unsqueeze(0)
        pred_index = torch.cat((output[0, 0, :, :][mask].unsqueeze(0), output[0, 1, :, :][mask].unsqueeze(0)), dim=0)
        gt_index = torch.cat((output2[0, 0, :, :][mask].unsqueeze(0), output2[0, 1, :, :][mask].unsqueeze(0)), dim=0)

        ## calculate normal loss
        H, W = depth_img.shape[2], depth_img.shape[3]
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, H, W)
        yy = yy.view(1, H, W)
        u0 = xx.to(device) - torch.tensor(W // 2, dtype=torch.float32).to(device)
        v0 = yy.to(device) - torch.tensor(H // 2, dtype=torch.float32).to(device)

        x = u0 * torch.abs(gt_depth_img_dilate) / fx
        y = v0 * torch.abs(gt_depth_img_dilate) / fy
        z = gt_depth_img_dilate
        pc_gt = torch.cat([x, y, z], 0).permute(1, 2, 0)
        x = u0 * torch.abs(pred_depth_img_dilate) / fx
        y = v0 * torch.abs(pred_depth_img_dilate) / fy
        z = pred_depth_img_dilate
        pc_pred = torch.cat([x, y, z], 0).permute(1, 2, 0)

        num = gt_index.shape[1]
        sample_num = int(0.1 * num)
        p1_index = np.random.choice(num, sample_num, replace=True)
        p2_index = np.random.choice(num, sample_num, replace=True)
        p3_index = np.random.choice(num, sample_num, replace=True)
        gt_p1_index = gt_index[:, p1_index].int()
        gt_p2_index = gt_index[:, p2_index].int()
        gt_p3_index = gt_index[:, p3_index].int()
        pred_p1_index = pred_index[:, p1_index].int()
        pred_p2_index = pred_index[:, p2_index].int()
        pred_p3_index = pred_index[:, p3_index].int()

        pc_gt_1 = pc_gt[gt_p1_index[0, :].cpu().numpy(), gt_p1_index[1, :].cpu().numpy(), :]
        pc_gt_2 = pc_gt[gt_p2_index[0, :].cpu().numpy(), gt_p2_index[1, :].cpu().numpy(), :]
        pc_gt_3 = pc_gt[gt_p3_index[0, :].cpu().numpy(), gt_p3_index[1, :].cpu().numpy(), :]
        pc_pred_1 = pc_pred[pred_p1_index[0, :].cpu().numpy(), pred_p1_index[1, :].cpu().numpy()]
        pc_pred_2 = pc_pred[pred_p2_index[0, :].cpu().numpy(), pred_p2_index[1, :].cpu().numpy()]
        pc_pred_3 = pc_pred[pred_p3_index[0, :].cpu().numpy(), pred_p3_index[1, :].cpu().numpy()]
        pc_gt_group = torch.cat([pc_gt_1[:, :, np.newaxis],
                                 pc_gt_2[:, :, np.newaxis],
                                 pc_gt_3[:, :, np.newaxis]], 2)       # Nx3x3
        pc_pred_group = torch.cat([pc_pred_1[:, :, np.newaxis],
                                   pc_pred_2[:, :, np.newaxis],
                                   pc_pred_3[:, :, np.newaxis]], 2)   # Nx3x3
        pc_gt_group.requires_grad = True
        pc_pred_group.requires_grad = True

        gt_p12 = pc_gt_group[:, :, 1] - pc_gt_group[:, :, 0]          # Nx3
        gt_p13 = pc_gt_group[:, :, 2] - pc_gt_group[:, :, 0]          # Nx3
        pred_p12 = pc_pred_group[:, :, 1] - pc_pred_group[:, :, 0]    # Nx3
        pred_p13 = pc_pred_group[:, :, 2] - pc_pred_group[:, :, 0]    # Nx3

        gt_normal = torch.cross(gt_p12, gt_p13, dim=1)
        gt_norm = torch.norm(gt_normal, 2, dim=1, keepdim=True)
        gt_normal = gt_normal / gt_norm
        pred_noraml = torch.cross(pred_p12, pred_p13, dim=1)
        pred_norm = torch.norm(pred_noraml, 2, dim=1, keepdim=True)
        pred_noraml = pred_noraml / pred_norm
        loss_p = torch.sum(torch.abs(gt_normal - pred_noraml), dim=1)
        loss_p = torch.mean(loss_p)

        loss = loss + loss_p

    return loss