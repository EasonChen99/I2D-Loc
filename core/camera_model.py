# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------
import numpy as np
import torch
import sys
sys.path.append("core")
from utils_point import rotate_forward, rotate_back

class CameraModel:

    def __init__(self, focal_length=None, principal_point=None):
        self.focal_length = focal_length
        self.principal_point = principal_point

    def project_pytorch(self, xyz: torch.Tensor, image_size, reflectance=None):
        if xyz.shape[0] == 3:
            xyz = torch.cat([xyz, torch.ones(1, xyz.shape[1], device=xyz.device)])
        else:
            if not torch.all(xyz[3, :] == 1.):
                xyz[3, :] = 1.
                raise TypeError("Wrong Coordinates")
        order = [1, 2, 0, 3]
        xyzw = xyz[order, :]
        indexes = xyzw[2, :] >= 0
        if reflectance is not None:
            reflectance = reflectance[:, indexes]
        xyzw = xyzw[:, indexes]

        uv = torch.zeros((2, xyzw.shape[1]), device=xyzw.device)
        uv[0, :] = self.focal_length[0] * xyzw[0, :] / xyzw[2, :] + self.principal_point[0]
        uv[1, :] = self.focal_length[1] * xyzw[1, :] / xyzw[2, :] + self.principal_point[1]
        indexes = uv[0, :] >= 0.1
        indexes = indexes & (uv[1, :] >= 0.1)
        indexes = indexes & (uv[0,:] < image_size[1])
        indexes = indexes & (uv[1,:] < image_size[0])
        if reflectance is None:
            uv = uv[:, indexes], xyzw[2, indexes], xyzw[:3, indexes], None
        else:
            uv = uv[:, indexes], xyzw[2, indexes], xyzw[:3, indexes], reflectance[:, indexes]

        return uv

    # for pc_RT
    def project_withindex_pytorch(self, xyz: torch.Tensor, image_size, reflectance=None):
        if xyz.shape[0] == 3:
            xyz = torch.cat([xyz, torch.ones(1, xyz.shape[1], device=xyz.device)])
        else:
            if not torch.all(xyz[3, :] == 1.):
                xyz[3, :] = 1.
                raise TypeError("Wrong Coordinates")
        order = [1, 2, 0, 3]
        xyzw = xyz[order, :]
        indexes = xyzw[2, :] >= 0
        if reflectance is not None:
            reflectance = reflectance[:, indexes]
        xyzw = xyzw[:, indexes]

        VI_indexes = indexes


        uv = torch.zeros((2, xyzw.shape[1]), device=xyzw.device)
        uv[0, :] = self.focal_length[0] * xyzw[0, :] / xyzw[2, :] + self.principal_point[0]
        uv[1, :] = self.focal_length[1] * xyzw[1, :] / xyzw[2, :] + self.principal_point[1]
        indexes = uv[0, :] >= 0.1
        indexes = indexes & (uv[1, :] >= 0.1)
        indexes = indexes & (uv[0, :] < image_size[1])
        indexes = indexes & (uv[1, :] < image_size[0])

        # generate complete indexes
        ind = torch.where(VI_indexes == True)[0]
        VI_indexes[ind] = VI_indexes[ind] & indexes


        if reflectance is None:
            uv = uv[:, indexes], xyzw[2, indexes], xyzw[:3, indexes], None, VI_indexes
        else:
            uv = uv[:, indexes], xyzw[2, indexes], xyzw[:3, indexes], reflectance[:, indexes], VI_indexes

        return uv


    def get_matrix(self):
        matrix = np.zeros([3, 3])
        matrix[0, 0] = self.focal_length[0]
        matrix[1, 1] = self.focal_length[1]
        matrix[0, 2] = self.principal_point[0]
        matrix[1, 2] = self.principal_point[1]
        matrix[2, 2] = 1.0
        return matrix

    def deproject_pytorch(self, uv, pc_project_uv):
        index = np.argwhere(uv > 0)
        mask = uv > 0
        z = uv[mask]
        x = (index[:, 1] - self.principal_point[0].cpu().numpy()) * z / self.focal_length[0].cpu().numpy()
        y = (index[:, 0] - self.principal_point[1].cpu().numpy()) * z / self.focal_length[1].cpu().numpy()
        zxy = np.array([z, x, y])
        zxy = torch.tensor(zxy, dtype=torch.float32).cuda()
        zxyw = torch.cat([zxy, torch.ones(1, zxy.shape[1], device=zxy.device)])
        # zxyw = rotate_forward(zxyw, R_ini, T_ini)
        zxy = zxyw[:3, :]
        zxy = zxy.cpu().numpy()
        xyz = zxy[[1, 2, 0], :]


        pc_project_u = pc_project_uv[:, :, 0][mask]
        pc_project_v = pc_project_uv[:, :, 1][mask]
        pc_project = np.array([pc_project_v, pc_project_u])


        match_index = np.array([index[:, 0], index[:, 1]])

        return xyz.transpose(), pc_project.transpose(), match_index.transpose()

    def depth2pc(self, depth_img):
        # pc = torch.zeros([depth_img.shape[0], depth_img.shape[1], 3])
        # pc[:, :, 2] = depth_img
        #
        # for i in range(depth_img.shape[0]):
        #     for j in range(depth_img.shape[1]):
        #         pc[i, j, 0] = (j - self.principal_point[0]) * depth_img[i, j] / self.focal_length[0]
        #         pc[i, j, 1] = (i - self.principal_point[1]) * depth_img[i, j] / self.focal_length[1]

        depth_img = depth_img.cpu().numpy()
        index = np.argwhere(depth_img > 0)
        mask = depth_img > 0
        z = depth_img[mask]
        x = (index[:, 1] - self.principal_point[0].cpu().numpy()) * z / self.focal_length[0].cpu().numpy()
        y = (index[:, 0] - self.principal_point[1].cpu().numpy()) * z / self.focal_length[1].cpu().numpy()

        # pc[index[:, 0], index[:, 1], 0] = x
        # pc[index[:, 0], index[:, 1], 1] = y
        # pc[index[:, 0], index[:, 1], 2] = z
        zxy = np.array([z, x, y], dtype=np.float32)
        return zxy