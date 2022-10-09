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

class VN_Loss(nn.Module):
    def __init__(self, focal_x, focal_y, input_size, device,
                 delta_cos=0.867, delta_diff_x=0.01,
                 delta_diff_y=0.01, delta_diff_z=0.01,
                 delta_z=0.0001, sample_ratio=0.15):
        super(VN_Loss, self).__init__()
        self.device = device
        self.fx = torch.tensor([focal_x], dtype=torch.float32).to(device)
        self.fy = torch.tensor([focal_y], dtype=torch.float32).to(device)
        self.input_size = input_size
        self.u0 = torch.tensor(input_size[1] // 2, dtype=torch.float32).to(device)
        self.v0 = torch.tensor(input_size[0] // 2, dtype=torch.float32).to(device)
        self.init_image_coor()
        self.delta_cos = delta_cos
        self.delta_diff_x = delta_diff_x
        self.delta_diff_y = delta_diff_y
        self.delta_diff_z = delta_diff_z
        self.delta_z = delta_z
        self.sample_ratio = sample_ratio

    def init_image_coor(self):
        x_row = np.arange(0, self.input_size[1])
        x = np.tile(x_row, (self.input_size[0], 1))
        x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy()).to(self.device)
        self.u_u0 = x - self.u0

        y_col = np.arange(0, self.input_size[0])
        y = np.tile(y_col, (self.input_size[1], 1)).T
        y = y[np.newaxis, :, :]
        y = y.astype(np.float32)
        y = torch.from_numpy(y.copy()).to(self.device)
        self.v_v0 = y - self.v0

    def transfer_xyz(self, depth):
        x = self.u_u0 * torch.abs(depth) / self.fx
        y = self.v_v0 * torch.abs(depth) / self.fy
        z = depth
        pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1)
        return pw

    def select_index(self, gt_index, pred_index):
        num = gt_index.shape[1]
        p1 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p1)
        p2 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p2)
        p3 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p3)

        gt_p1_x = gt_index[1][p1].astype(np.int)
        gt_p1_y = gt_index[0][p1].astype(np.int)
        pred_p1_x = pred_index[1][p1].astype(np.int)
        pred_p1_y = pred_index[0][p1].astype(np.int)

        gt_p2_x = gt_index[1][p2].astype(np.int)
        gt_p2_y = gt_index[0][p2].astype(np.int)
        pred_p2_x = pred_index[1][p2].astype(np.int)
        pred_p2_y = pred_index[0][p2].astype(np.int)

        gt_p3_x = gt_index[1][p3].astype(np.int)
        gt_p3_y = gt_index[0][p3].astype(np.int)
        pred_p3_x = pred_index[1][p3].astype(np.int)
        pred_p3_y = pred_index[0][p3].astype(np.int)

        gt_p123 = {'p1_x': gt_p1_x, 'p1_y': gt_p1_y, 'p2_x': gt_p2_x, 'p2_y': gt_p2_y, 'p3_x': gt_p3_x, 'p3_y': gt_p3_y}
        pred_p123 = {'p1_x': pred_p1_x, 'p1_y': pred_p1_y, 'p2_x': pred_p2_x, 'p2_y': pred_p2_y, 'p3_x': pred_p3_x, 'p3_y': pred_p3_y}
        return gt_p123, pred_p123

    def form_pw_groups(self, p123, pw):
        p1_x = p123['p1_x']
        p1_y = p123['p1_y']
        p2_x = p123['p2_x']
        p2_y = p123['p2_y']
        p3_x = p123['p3_x']
        p3_y = p123['p3_y']

        pw1 = pw[:, p1_y, p1_x, :]
        pw2 = pw[:, p2_y, p2_x, :]
        pw3 = pw[:, p3_y, p3_x, :]

        pw_groups = torch.cat([pw1[:, :, :, np.newaxis], pw2[:, :, :, np.newaxis], pw3[:, :, :, np.newaxis]], 3)
        return pw_groups

    def filter_mask(self, p123, gt_xyz, pred_xyz, delta_cos=0.867,
                    delta_diff_x=0.005,
                    delta_diff_y=0.005,
                    delta_diff_z=0.005):
        pw = self.form_pw_groups(p123, gt_xyz)
        pw_pred = self.form_pw_groups(p123, pred_xyz)

        ## common
        mask_com = (torch.sum(pw[:, :, 2, :] > self.delta_z, 2) == 3) & (torch.sum(pw_pred[:, :, 2, :] > self.delta_z, 2) == 3)

        pw12 = pw[:, :, :, 1] - pw[:, :, :, 0]
        pw13 = pw[:, :, :, 2] - pw[:, :, :, 0]
        pw23 = pw[:, :, :, 2] - pw[:, :, :, 1]
        ## ignore linear
        pw_diff = torch.cat([pw12[:, :, :, np.newaxis], pw13[:, :, :, np.newaxis], pw23[:, :, :, np.newaxis]],
                            3)  # [b, n, 3, 3]
        m_batchsize, groups, coords, index = pw_diff.shape
        proj_query = pw_diff.view(m_batchsize * groups, -1, index).permute(0, 2, 1)
        proj_key = pw_diff.view(m_batchsize * groups, -1, index)
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.view(m_batchsize * groups, index, 1), q_norm.view(m_batchsize * groups, 1, index)) #[]
        energy = torch.bmm(proj_query, proj_key)
        norm_energy = energy / (nm + 1e-8)
        norm_energy = norm_energy.view(m_batchsize * groups, -1)
        mask_cos = torch.sum((norm_energy > delta_cos) + (norm_energy < -delta_cos), 1) > 3
        mask_cos = mask_cos.view(m_batchsize, groups)

        ## ignore padding and invilid depth
        mask_pad = torch.sum(pw[:, :, 2, :] > self.delta_z, 2) == 3

        ## ignore near
        mask_x = torch.sum(torch.abs(pw_diff[:, :, 0, :]) < delta_diff_x, 2) > 0
        mask_y = torch.sum(torch.abs(pw_diff[:, :, 1, :]) < delta_diff_y, 2) > 0
        mask_z = torch.sum(torch.abs(pw_diff[:, :, 2, :]) < delta_diff_z, 2) > 0

        mask_ignore = (mask_x & mask_y & mask_z) | mask_cos
        mask_near = ~mask_ignore
        mask = mask_pad & mask_near

        mask = mask & mask_com

        return mask, pw

    def select_points_groups(self, gt_depth, pred_depth, gt_index, pred_index):
        pw_gt = self.transfer_xyz(gt_depth)
        pw_pred = self.transfer_xyz(pred_depth)

        gt_p123, pred_p123 = self.select_index(gt_index, pred_index)

        pw_groups_gt = self.form_pw_groups(gt_p123, pw_gt)
        ## [b, n, 3, 3]
        pw_groups_pred = self.form_pw_groups(pred_p123, pw_pred)
        pw_groups_pred_not_ignore = pw_groups_pred.reshape(1, -1, 3, 3)
        pw_groups_gt_not_ignore = pw_groups_gt.reshape(1, -1, 3, 3)

        return pw_groups_gt_not_ignore, pw_groups_pred_not_ignore

    def forward(self, gt_depth, pred_depth, gt_index, pred_index, select=True):
        gt_index = gt_index.cpu().numpy()
        pred_index = pred_index.cpu().numpy()
        gt_points, dt_points = self.select_points_groups(gt_depth, pred_depth, gt_index, pred_index)
        gt_points.requires_grad = True
        dt_points.requires_grad = True

        gt_p12 = gt_points[:, :, :, 1] - gt_points[:, :, :, 0]
        gt_p13 = gt_points[:, :, :, 2] - gt_points[:, :, :, 0]
        dt_p12 = dt_points[:, :, :, 1] - dt_points[:, :, :, 0]
        dt_p13 = dt_points[:, :, :, 2] - dt_points[:, :, :, 0]

        gt_normal = torch.cross(gt_p12, gt_p13, dim=2)
        dt_normal = torch.cross(dt_p12, dt_p13, dim=2)
        dt_norm = torch.norm(dt_normal, 2, dim=2, keepdim=True)
        gt_norm = torch.norm(gt_normal, 2, dim=2, keepdim=True)
        dt_mask = dt_norm == 0.0
        gt_mask = gt_norm == 0.0
        dt_mask = dt_mask.to(torch.float32)
        gt_mask = gt_mask.to(torch.float32)
        dt_mask *= 0.01
        gt_mask *= 0.01
        gt_norm = gt_norm + gt_mask
        dt_norm = dt_norm + dt_mask
        gt_normal = gt_normal / gt_norm
        dt_normal = dt_normal / dt_norm
        loss = torch.abs(gt_normal - dt_normal)
        loss = torch.sum(torch.sum(loss, dim=2), dim=0)
        if select:
            loss, indices = torch.sort(loss, dim=0, descending=False)
            loss = loss[int(loss.size(0) * 0.25):]
        loss = torch.mean(loss)
        return loss



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
        pred_depth_img_dilate = sparse_to_dense(pred_depth_img[0, 0, :, :].cpu().numpy().astype(np.float32))

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

        vnl_loss = VN_Loss(cam_params[0], cam_params[1],
                           [depth_img.shape[2], depth_img.shape[3]], device)
        loss_p = vnl_loss.forward(torch.tensor(gt_depth_img_dilate).to(device).unsqueeze(0).unsqueeze(0),
                                  torch.tensor(pred_depth_img_dilate).to(device).unsqueeze(0).unsqueeze(0),
                                  gt_index, pred_index)
        loss += loss_p

    return loss