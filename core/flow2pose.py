import torch
import visibility
import cv2
import numpy as np
import mathutils
import math

import sys
sys.path.append('core')
from camera_model import CameraModel
from utils_point import invert_pose, quat2mat, tvector2mat, quaternion_from_matrix
from quaternion_distances import quaternion_distance

def Flow2Pose(flow_up, depth, calib):
    device = flow_up.device

    output = torch.zeros(flow_up.shape).to(device)
    pred_depth_img = torch.zeros(depth.shape).to(device)
    pred_depth_img += 1000.
    output = visibility.image_warp_index(depth.to(device), flow_up.int(), pred_depth_img, output,
                                         depth.shape[3], depth.shape[2])
    pred_depth_img[pred_depth_img == 1000.] = 0.
    pc_project_uv = output.cpu().permute(0, 2, 3, 1).numpy()

    depth_img_ori = depth.cpu().numpy() * 100.

    mask_depth_1 = pc_project_uv[0, :, :, 0] != 0
    mask_depth_2 = pc_project_uv[0, :, :, 1] != 0
    mask_depth = mask_depth_1 + mask_depth_2
    depth_img = depth_img_ori[0, 0, :, :] * mask_depth

    cam_model = CameraModel()
    cam_params = calib[0].cpu().numpy()
    x, y = 28, 140
    cam_params[2] = cam_params[2] + 480 - (y + y + 960) / 2.
    cam_params[3] = cam_params[3] + 160 - (x + x + 320) / 2.
    cam_model.focal_length = cam_params[:2]
    cam_model.principal_point = cam_params[2:]
    cam_mat = np.array([[cam_params[0], 0, cam_params[2]], [0, cam_params[1], cam_params[3]], [0, 0, 1.]])

    pts3d, pts2d, match_index = cam_model.deproject_pytorch(depth_img, pc_project_uv[0, :, :, :])
    ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(pts3d, pts2d, cam_mat, None)

    R = mathutils.Euler((rvecs[0], rvecs[1], rvecs[2]), 'XYZ')
    T = mathutils.Vector((tvecs[0], tvecs[1], tvecs[2]))
    R, T = invert_pose(R, T)
    R, T = torch.tensor(R), torch.tensor(T)
    R_predicted = R[[0, 3, 1, 2]]
    T_predicted = T[[2, 0, 1]]

    return R_predicted, T_predicted

def Flow2PoseBPnP(flow_up, depth, calib, bpnp):
    device =flow_up.device
    output = torch.zeros(flow_up.shape).to(device)
    pred_depth_img = torch.zeros(depth.shape).to(device)
    pred_depth_img += 1000.
    output = visibility.image_warp_index(depth.to(device), flow_up.int().to(device), pred_depth_img, output,
                                         depth.shape[3], depth.shape[2])
    pred_depth_img[pred_depth_img == 1000.] = 0.
    pc_project_uv = output.cpu().permute(0, 2, 3, 1).numpy()

    depth_img_ori = depth.cpu().numpy() * 100.

    mask_depth_1 = pc_project_uv[0, :, :, 0] != 0
    mask_depth_2 = pc_project_uv[0, :, :, 1] != 0
    mask_depth = mask_depth_1 + mask_depth_2
    depth_img = depth_img_ori[0, 0, :, :] * mask_depth

    cam_model = CameraModel()
    cam_params = calib[0].cpu().numpy()
    x, y = 28, 140
    cam_params[2] = cam_params[2] + 480 - (y + y + 960) / 2.
    cam_params[3] = cam_params[3] + 160 - (x + x + 320) / 2.
    cam_model.focal_length = cam_params[:2]
    cam_model.principal_point = cam_params[2:]
    cam_mat = np.array([[cam_params[0], 0, cam_params[2]], [0, cam_params[1], cam_params[3]], [0, 0, 1.]])

    pts3d, pts2d, match_index = cam_model.deproject_pytorch(depth_img, pc_project_uv[0, :, :, :])

    pts3d = torch.tensor(pts3d, dtype=torch.float32).to(device)
    pts2d = torch.tensor(pts2d, dtype=torch.float32).to(device)
    pts2d = pts2d.unsqueeze(0)
    K = torch.tensor(cam_mat, dtype=torch.float32).to(device)
    P_out = bpnp(pts2d, pts3d, K)
    rvecs = P_out[0, 0:3]
    tvecs = P_out[0, 3:]

    R = mathutils.Euler((rvecs[0], rvecs[1], rvecs[2]), 'XYZ')
    T = mathutils.Vector((tvecs[0], tvecs[1], tvecs[2]))
    R, T = invert_pose(R, T)
    R, T = torch.tensor(R), torch.tensor(T)
    R_predicted = R[[0, 3, 1, 2]]
    T_predicted = T[[2, 0, 1]]

    return R_predicted, T_predicted

def err_Pose(R_pred, T_pred, R_gt, T_gt):
    device = R_pred.device

    R = quat2mat(R_gt)
    T = tvector2mat(T_gt)
    RT_inv = torch.mm(T, R).to(device)
    RT = RT_inv.clone().inverse()

    R_pred = quat2mat(R_pred)
    T_pred = tvector2mat(T_pred)
    RT_pred = torch.mm(T_pred, R_pred)
    RT_pred = RT_pred.to(device)
    RT_new = torch.mm(RT, RT_pred)

    T_composed = RT_new[:3, 3]
    R_composed = quaternion_from_matrix(RT_new)
    R_composed = R_composed.unsqueeze(0)
    total_trasl_error = torch.tensor(0.0).to(device)
    total_rot_error = quaternion_distance(R_composed.to(device), torch.tensor([[1., 0., 0., 0.]]).to(device),
                                          device=R_composed.device)
    total_rot_error = total_rot_error * 180. / math.pi
    total_trasl_error += torch.norm(T_composed.to(device)) * 100.

    total_trasl_fail = torch.norm(T_composed - T_gt[0].to(device)) * 100
    if total_trasl_fail > 400:
        is_fail = True
    else:
        is_fail = False
    return total_rot_error, total_trasl_error, is_fail


