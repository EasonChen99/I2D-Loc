import torch
import visibility
import mathutils
import numpy as np

import sys
sys.path.append('core')
from utils_point import overlay_imgs, rotate_back
from camera_model import CameraModel

class Data_preprocess:
    def __init__(self, calibs, occlusion_threshold, occlusion_kernel):
        self.real_shape = None
        self.calibs = calibs
        self.occlusion_threshold = occlusion_threshold
        self.occlusion_kernel = occlusion_kernel

    def delta_1(self, uv_RT, uv, VI_indexes_RT, VI_indexes):
        indexes = VI_indexes_RT & VI_indexes

        indexes_1 = indexes[VI_indexes_RT]
        indexes_2 = indexes[VI_indexes]

        delta_P = uv[indexes_2, :] - uv_RT[indexes_1, :]

        return delta_P, indexes

    def gen_depth_img(self, uv_RT_af_index, depth_RT_af_index, indexes_uvRT, cam_params):
        device = uv_RT_af_index.device

        depth_img_RT = torch.zeros(self.real_shape[:2], device=device, dtype=torch.float)
        depth_img_RT += 1000.

        idx_img = (-1) * torch.ones(self.real_shape[:2], device=device, dtype=torch.float)
        indexes_uvRT = indexes_uvRT.float()

        depth_img_RT, idx_img = visibility.depth_image(uv_RT_af_index, depth_RT_af_index, indexes_uvRT,
                                                       depth_img_RT, idx_img, uv_RT_af_index.shape[0],
                                                       self.real_shape[1], self.real_shape[0])
        depth_img_RT[depth_img_RT == 1000.] = 0.

        deoccl_index_img = (-1) * torch.ones(self.real_shape[:2], device=device, dtype=torch.float)

        depth_img_no_occlusion_RT = torch.zeros_like(depth_img_RT, device=device)
        depth_img_no_occlusion_RT, deoccl_index_img = visibility.visibility2(depth_img_RT, cam_params,
                                                                             idx_img,
                                                                             depth_img_no_occlusion_RT,
                                                                             deoccl_index_img,
                                                                             depth_img_RT.shape[1],
                                                                             depth_img_RT.shape[0],
                                                                             self.occlusion_threshold,
                                                                             self.occlusion_kernel)

        return depth_img_no_occlusion_RT, deoccl_index_img.int()

    def fresh_indexes(self, indexes_uvRT_deoccl, indexes_uvRT):
        indexes_uvRT_deoccl_list_indexes = torch.where(indexes_uvRT_deoccl > 0)

        indexes_uvRT_deoccl_list = indexes_uvRT_deoccl[indexes_uvRT_deoccl_list_indexes[0][:], indexes_uvRT_deoccl_list_indexes[1][:]]

        indexes_temp = torch.zeros(indexes_uvRT.shape[0], device=indexes_uvRT_deoccl_list.device, dtype=torch.int32)
        indexes_temp[indexes_uvRT_deoccl_list.cpu().numpy() - 1] = indexes_uvRT_deoccl_list

        return indexes_temp

    def delta_2(self, delta_P, uv_RT_af_index, mask):
        device = delta_P.device

        delta_P_com = delta_P[mask, :]

        delta_P_0 = delta_P_com[:, 0]
        delta_P_1 = delta_P_com[:, 1]

        ## keep common points after deocclusion
        uv_RT_af_index_com = uv_RT_af_index[mask, :]

        ## generate displacement map
        project_delta_P_1 = torch.zeros(self.real_shape[:2], device=device, dtype=torch.int32)
        project_delta_P_2 = torch.zeros(self.real_shape[:2], device=device, dtype=torch.int32)
        project_delta_P_1[uv_RT_af_index_com[:, 1].cpu().numpy(), uv_RT_af_index_com[:, 0].cpu().numpy()] = delta_P_0
        project_delta_P_2[uv_RT_af_index_com[:, 1].cpu().numpy(), uv_RT_af_index_com[:, 0].cpu().numpy()] = delta_P_1

        project_delta_P_shape = list(self.real_shape[:2])
        project_delta_P_shape.insert(0, 2)
        project_delta_P = torch.zeros(project_delta_P_shape, device=device, dtype=torch.float)

        project_delta_P[0, :, :] = project_delta_P_1
        project_delta_P[1, :, :] = project_delta_P_2

        return project_delta_P

    def DownsampleCrop_KITTI_delta(self, img, depth, displacement, split):
        if split == 'train':
            x = np.random.randint(0, img.shape[1] - 320)
            y = np.random.randint(0, img.shape[2] - 960)
        else:
            x = (img.shape[1] - 320) // 2
            y = (img.shape[2] - 960) // 2
        img = img[:, x:x + 320, y:y + 960]
        depth = depth[:, x:x + 320, y:y + 960]
        displacement = displacement[:, x:x + 320, y:y + 960]
        return img, depth, displacement


    def push(self, rgbs, pcs, T_errs, R_errs, device, split='train'):
        lidar_input = []
        rgb_input = []
        flow_gt = []

        for idx in range(len(rgbs)):
            rgb = rgbs[idx].to(device)
            pc = pcs[idx].clone().to(device)
            reflectance = None

            self.real_shape = [rgb.shape[1], rgb.shape[2], rgb.shape[0]]

            R = mathutils.Quaternion(R_errs[idx].to(device)).to_matrix()
            R.resize_4x4()
            T = mathutils.Matrix.Translation(T_errs[idx].to(device))
            RT = T * R

            pc_rotated = rotate_back(pc, RT)

            cam_params = self.calibs[idx]
            cam_model = CameraModel()
            cam_model.focal_length = cam_params[:2]
            cam_model.principal_point = cam_params[2:]
            cam_params = cam_params.to(device)

            uv, depth, _, refl, VI_indexes = cam_model.project_withindex_pytorch(pc, self.real_shape, reflectance)
            uv = uv.t().int().contiguous()

            uv_RT, depth_RT, _, refl_RT, VI_indexes_RT = cam_model.project_withindex_pytorch(pc_rotated, self.real_shape,
                                                                                             reflectance)
            uv_RT = uv_RT.t().int().contiguous()

            delta_P, indexes = self.delta_1(uv_RT, uv, VI_indexes_RT, VI_indexes)

            indexes_uvRT = VI_indexes_RT[indexes]
            indexes_uvRT = torch.arange(indexes_uvRT.shape[0]).to(device) + 1

            ## keep common points
            uv_RT_af_index = uv_RT[indexes[VI_indexes_RT], :]
            depth_RT_af_index = depth_RT[indexes[VI_indexes_RT]]

            indexes_uv = VI_indexes[indexes]
            indexes_uv = torch.arange(indexes_uv.shape[0]).to(device) + 1

            ## keep common points
            uv_af_index = uv[indexes[VI_indexes], :]
            depth_af_index = depth[indexes[VI_indexes]]

            depth_img_no_occlusion_RT, indexes_uvRT_deoccl = self.gen_depth_img(uv_RT_af_index, depth_RT_af_index,
                                                                                   indexes_uvRT, cam_params)
            indexes_uvRT_fresh = self.fresh_indexes(indexes_uvRT_deoccl, indexes_uvRT)

            depth_img_no_occlusion, indexes_uv_deoccl = self.gen_depth_img(uv_af_index, depth_af_index, indexes_uv, cam_params)
            indexes_uv_fresh = self.fresh_indexes(indexes_uv_deoccl, indexes_uv)

            ## make depth_image for training
            depth_img_no_occlusion_RT_training, indexes_uvRT_deoccl_training = \
                self.gen_depth_img(uv_RT, depth_RT, VI_indexes_RT[VI_indexes_RT], cam_params)

            ## 这里归一化的时候是不是重新计算一下最大深度比较好
            depth_img_no_occlusion_RT_training /= 100.

            depth_img_no_occlusion_RT_training = depth_img_no_occlusion_RT_training.unsqueeze(0)

            mask1 = indexes_uv_fresh > 0
            mask2 = indexes_uvRT_fresh > 0
            mask = mask1 & mask2
            project_delta_P = self.delta_2(delta_P, uv_RT_af_index, mask)

            ## downsample and crop
            rgb, depth_img_no_occlusion_RT_training, project_delta_P \
                = self.DownsampleCrop_KITTI_delta(rgb, depth_img_no_occlusion_RT_training, project_delta_P, split)

            rgb_input.append(rgb)
            lidar_input.append(depth_img_no_occlusion_RT_training)
            flow_gt.append(project_delta_P)

        lidar_input = torch.stack(lidar_input)
        rgb_input = torch.stack(rgb_input)
        flow_gt = torch.stack(flow_gt)

        return rgb_input, lidar_input, flow_gt