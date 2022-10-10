import sys
import os
import numpy as np
import h5py
import argparse
import torch
from torchvision import transforms
import mathutils
from PIL import Image
import cv2
import visibility
from core.raft import RAFT
from core.utils_point import invert_pose, overlay_imgs
from core.data_preprocess import Data_preprocess
from core.depth_completion import sparse_to_dense
from core.flow_viz import flow_to_image
from core.flow2pose import Flow2Pose, err_Pose

def custom_transform(rgb):
    to_tensor = transforms.ToTensor()
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    rgb = to_tensor(rgb)
    rgb = normalization(rgb)

    return rgb

def load_data(root, id):
    img_path = os.path.join(root, "image", id + '.png')
    pc_path = os.path.join(root, "pc", id + '.h5')

    try:
        with h5py.File(pc_path, 'r') as hf:
            pc = hf['PC'][:]
    except Exception as e:
        print(f'File Broken: {pc_path}')
        raise e

    pc_in = torch.from_numpy(pc.astype(np.float32))
    if pc_in.shape[1] == 4 or pc_in.shape[1] == 3:
        pc_in = pc_in.t()
    if pc_in.shape[0] == 3:
        homogeneous = torch.ones(pc_in.shape[1]).unsqueeze(0)
        pc_in = torch.cat((pc_in, homogeneous), 0)
    elif pc_in.shape[0] == 4:
        if not torch.all(pc_in[3, :] == 1.):
            pc_in[3, :] = 1.
    else:
        raise TypeError("Wrong PointCloud shape")

    img = Image.open(img_path)
    img = custom_transform(img)

    max_r = 10.
    max_t = 2.
    max_angle = max_r
    rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
    roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
    rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
    transl_x = np.random.uniform(-max_t, max_t)
    transl_y = np.random.uniform(-max_t, max_t)
    transl_z = np.random.uniform(-max_t, min(max_t, 1.))

    R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
    T = mathutils.Vector((transl_x, transl_y, transl_z))

    R, T = invert_pose(R, T)
    R, T = torch.tensor(R), torch.tensor(T)

    return pc_in, img, R, T


def demo(args):
    device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.set_device(args.gpus[0])

    root = args.data_path
    calib = torch.tensor([718.856, 718.856, 607.1928, 185.2157])
    calib = calib.unsqueeze(0)
    occlusion_kernel = 5
    occlusion_threshold = 3
    id_list = sorted(os.listdir(os.path.join(root, "image")))
    id_list = [id[:6] for id in id_list]

    model = torch.nn.DataParallel(RAFT(args), device_ids=args.gpus)
    model.load_state_dict(torch.load(args.load_checkpoints))
    model.to(device)

    for k in range(len(id_list)):
        id = id_list[k]

        pc, img, R_err, T_err = load_data(root, id)
        pc = pc.unsqueeze(0)
        img = img.unsqueeze(0)
        R_err = R_err.unsqueeze(0)
        T_err = T_err.unsqueeze(0)
        calib_k = calib.clone()

        data_generate = Data_preprocess(calib_k, occlusion_threshold, occlusion_kernel)
        rgb_input, spare_depth, flow_gt = data_generate.push(img, pc, T_err, R_err, device, split='test')

        # dilation
        dense_depth = []
        for i in range(spare_depth.shape[0]):
            depth_img = spare_depth[i, 0, :, :].cpu().numpy() * 100.
            depth_img_dilate = sparse_to_dense(depth_img.astype(np.float32))
            dense_depth.append(depth_img_dilate / 100.)
        dense_depth = torch.tensor(np.array(dense_depth)).float().to(device)
        dense_depth = dense_depth.unsqueeze(1)

        _, flow_up = model(dense_depth, rgb_input, lidar_mask=spare_depth, iters=24, test_mode=True)

        if args.render:
            if not os.path.exists(f"{root}/visualization"):
                os.mkdir(f"{root}/visualization")
                os.mkdir(f"{root}/visualization/flow")
                os.mkdir(f"{root}/visualization/original_overlay")
                os.mkdir(f"{root}/visualization/warp_overlay")

            flow_image = flow_to_image(flow_up.permute(0, 2, 3, 1).cpu().detach().numpy()[0])
            cv2.imwrite(f'{root}/visualization/flow/{id}.png', flow_image)

            output = torch.zeros(flow_up.shape).to(device)
            pred_depth_img = torch.zeros(spare_depth.shape).to(device)
            pred_depth_img += 1000.
            output = visibility.image_warp_index(spare_depth.to(device),
                                                 flow_up.int().to(device), pred_depth_img,
                                                 output, spare_depth.shape[3], spare_depth.shape[2])
            pred_depth_img[pred_depth_img == 1000.] = 0.

            original_overlay = overlay_imgs(rgb_input[0, :, :, :], spare_depth[0, 0, :, :])
            cv2.imwrite(f'{root}/visualization/original_overlay/{id}.png', original_overlay)
            warp_overlay = overlay_imgs(rgb_input[0, :, :, :], pred_depth_img[0, 0, :, :])
            cv2.imwrite(f'{root}/visualization/warp_overlay/{id}.png', warp_overlay)

        R_pred, T_pred = Flow2Pose(flow_up, spare_depth, calib_k)
        R_gt = torch.tensor([1., 0., 0., 0.])
        T_gt = torch.tensor([0., 0., 0.])
        init_err_r, init_err_t, _ = err_Pose(R_err[0], T_err[0], R_gt, T_gt)
        pred_err_r, pred_err_t, _ = err_Pose(R_pred, T_pred, R_err[0], T_err[0])
        print(f"sample {id}:")
        print(f"initial rotation error {init_err_r.item():.5f}  initial translation error {init_err_t.item():.5f} cm")
        print(f"prediction rotation error {pred_err_r.item():.5f}  prediction translation error {pred_err_t.item():.5f} cm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, metavar='DIR', default="./sample", help='path to dataset')
    parser.add_argument('-cps', '--load_checkpoints', help="restore checkpoint")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    demo(args)