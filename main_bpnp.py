import sys
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
import argparse
import visibility
import time

sys.path.append('core')
from raft import RAFT
from datasets_kitti import DatasetVisibilityKittiSingle
from camera_model import CameraModel
from utils import fetch_optimizer, Logger, count_parameters
from utils_point import merge_inputs, overlay_imgs, quat2mat, tvector2mat, mat2xyzrpy
from data_preprocess import Data_preprocess
from losses import sequence_loss, normal_loss
from depth_completion import sparse_to_dense
from flow_viz import flow_to_image
from flow2pose import Flow2PoseBPnP, err_Pose
from BPnP import BPnP, batch_project

occlusion_kernel = 5  # 3
occlusion_threshold = 3  # 3.9999
seed = 1234
BPnP_EPOCH = 30

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

def _init_fn(worker_id, seed):
    seed = seed
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    np.random.seed(seed)

def train(args, epoch, TrainImgLoader, model, optimizer, scheduler, scaler, logger, device):
    global occlusion_threshold, occlusion_kernel
    model.train()
    bpnp = BPnP.apply
    cam_model = CameraModel()
    for i_batch, sample in enumerate(TrainImgLoader):
        rgb = sample['rgb']
        pc = sample['point_cloud']
        calib = sample['calib']
        T_err = sample['tr_error']
        R_err = sample['rot_error']

        data_generate = Data_preprocess(calib, occlusion_threshold, occlusion_kernel)
        rgb_input, lidar_input, flow_gt = data_generate.push(rgb, pc, T_err, R_err, device)

        # dilation
        depth_img_input = []
        for i in range(lidar_input.shape[0]):
            depth_img = lidar_input[i, 0, :, :].cpu().numpy() * 100.
            depth_img_dilate = sparse_to_dense(depth_img.astype(np.float32))
            depth_img_input.append(depth_img_dilate / 100.)
        depth_img_input = torch.tensor(depth_img_input).float().to(device)
        depth_img_input = depth_img_input.unsqueeze(1)

        optimizer.zero_grad()
        flow_preds = model(depth_img_input, rgb_input, lidar_mask=lidar_input, iters=args.iters)

        loss, metrics = sequence_loss(flow_preds, flow_gt, args.gamma, MAX_FLOW=400)
        norm_loss = normal_loss(flow_preds, flow_gt, calib, lidar_input)
        loss += norm_loss * 100

        ## BPnP loss
        if epoch > BPnP_EPOCH:
            loss_poses = 0.0

            flow_up = flow_preds[-1]

            depth_img_ori = lidar_input.cpu().numpy() * 100.
            pc_project_uv = np.zeros([lidar_input.shape[0], lidar_input.shape[2], lidar_input.shape[3], 2])

            for i in range(flow_up.shape[0]):
                output = torch.zeros([1, flow_up.shape[1], flow_up.shape[2], flow_up.shape[3]]).to(device)
                warp_depth_img = torch.zeros([1, 1, flow_up.shape[2], flow_up.shape[3]]).to(device)
                warp_depth_img += 1000.
                output = visibility.image_warp_index(lidar_input[i, :, :, :].unsqueeze(0).to(device),
                                                     flow_up[i, :, :, :].unsqueeze(0).int().to(device), warp_depth_img,
                                                     output, lidar_input.shape[3], lidar_input.shape[2])
                warp_depth_img[warp_depth_img == 1000.] = 0
                pc_project_uv[i, :, :, :] = output.cpu().permute(0, 2, 3, 1).numpy()

            for n in range(lidar_input.shape[0]):
                mask_depth_1 = pc_project_uv[n, :, :, 0] != 0
                mask_depth_2 = pc_project_uv[n, :, :, 1] != 0
                mask_depth = mask_depth_1 + mask_depth_2
                depth_img = depth_img_ori[n, 0, :, :] * mask_depth
                cam_params_clone = calib[n].cpu().numpy()
                cam_model.focal_length = cam_params_clone[:2]
                cam_model.principal_point = cam_params_clone[2:]
                pts3d, pts2d, _ = cam_model.deproject_pytorch(depth_img, pc_project_uv[n, :, :, :])
                pts3d = torch.tensor(pts3d, dtype=torch.float32).to(device)
                pts2d = torch.tensor(pts2d, dtype=torch.float32).to(device)
                pts2d = pts2d.unsqueeze(0)
                cam_mat = np.array(
                    [[cam_params_clone[0], 0, cam_params_clone[2]], [0, cam_params_clone[1], cam_params_clone[3]],
                     [0, 0, 1.]])
                K = torch.tensor(cam_mat, dtype=torch.float32).to(device)

                P_out = bpnp(pts2d, pts3d, K)
                pts2d_pro = batch_project(P_out, pts3d, K)

                R = quat2mat(R_err[n])
                T = tvector2mat(T_err[n])
                RT_inv = torch.mm(T, R)
                RT = RT_inv.clone().inverse()
                P_gt = mat2xyzrpy(RT)
                P_gt = P_gt[[4, 5, 3, 1, 2, 0]]
                P_gt = P_gt.unsqueeze(0)
                pts2d_gt = batch_project(P_gt.to(device), pts3d, K)

                pts2d_gt.requires_grad = True
                pts2d_pro.requires_grad = True
                loss_poses += ((pts2d_pro - pts2d_gt) ** 2).mean()

            loss_pose = loss_poses / lidar_input.shape[0]
        else:
            loss_pose = 0.

        loss += loss_pose * 10

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        scaler.step(optimizer)
        scheduler.step()
        scaler.update()

        logger.push(metrics)

def test(args, TestImgLoader, model, bpnp, device, cal_pose=False):
    global occlusion_threshold, occlusion_kernel
    model.eval()
    out_list, epe_list = [], []
    Time = 0.
    outliers, err_r_list, err_t_list = [], [], []
    for i_batch, sample in enumerate(TestImgLoader):
        rgb = sample['rgb']
        pc = sample['point_cloud']
        calib = sample['calib']
        T_err = sample['tr_error']
        R_err = sample['rot_error']

        data_generate = Data_preprocess(calib, occlusion_threshold, occlusion_kernel)
        rgb_input, lidar_input, flow_gt = data_generate.push(rgb, pc, T_err, R_err, device, split='test')

        # dilation
        depth_img_input = []
        for i in range(lidar_input.shape[0]):
            depth_img = lidar_input[i, 0, :, :].cpu().numpy() * 100.
            depth_img_dilate = sparse_to_dense(depth_img.astype(np.float32))
            depth_img_input.append(depth_img_dilate / 100.)
        depth_img_input = torch.tensor(depth_img_input).float().to(device)
        depth_img_input = depth_img_input.unsqueeze(1)

        end = time.time()
        _, flow_up = model(depth_img_input, rgb_input, lidar_mask=lidar_input, iters=24, test_mode=True)

        if args.render:
            if not os.path.exists(f"./visulization"):
                os.mkdir(f"./visulization")
                os.mkdir(f"./visulization/flow")
                os.mkdir(f"./visulization/original_overlay")
                os.mkdir(f"./visulization/warp_overlay")

            flow_image = flow_to_image(flow_up.permute(0, 2, 3, 1).cpu().detach().numpy()[0])
            cv2.imwrite(f'./visulization/flow/{i_batch:06d}.png', flow_image)

            output = torch.zeros(flow_up.shape).to(device)
            pred_depth_img = torch.zeros(lidar_input.shape).to(device)
            pred_depth_img += 1000.
            output = visibility.image_warp_index(lidar_input.to(device),
                                                 flow_up.int().to(device), pred_depth_img,
                                                 output, lidar_input.shape[3], lidar_input.shape[2])
            pred_depth_img[pred_depth_img == 1000.] = 0.

            original_overlay = overlay_imgs(rgb_input[0, :, :, :], lidar_input[0, 0, :, :])
            cv2.imwrite(f'./visulization/original_overlay/{i_batch:06d}.png', original_overlay)
            warp_overlay = overlay_imgs(rgb_input[0, :, :, :], pred_depth_img[0, 0, :, :])
            cv2.imwrite(f'./visulization/warp_overlay/{i_batch:06d}.png', warp_overlay)

        if not cal_pose:
            epe = torch.sum((flow_up - flow_gt) ** 2, dim=1).sqrt()
            mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
            epe = epe.view(-1)
            mag = mag.view(-1)
            valid_gt = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)
            val = valid_gt.view(-1) >= 0.5

            out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
            epe_list.append(epe[val].mean().item())
            out_list.append(out[val].cpu().numpy())
        else:
            R_pred, T_pred = Flow2PoseBPnP(flow_up, lidar_input, calib, bpnp)
            Time += time.time() - end
            err_r, err_t, is_fail = err_Pose(R_pred, T_pred, R_err[0], T_err[0])
            if is_fail:
                outliers.append(i_batch)
            else:
                err_r_list.append(err_r.item())
                err_t_list.append(err_t.item())
            print(f"{i_batch:05d}: {np.mean(err_t_list):.5f} {np.mean(err_r_list):.5f} {np.median(err_t_list):.5f} "
                  f"{np.median(err_r_list):.5f} {len(outliers)} {Time / (i_batch+1):.5f}")

    if not cal_pose:
        epe_list = np.array(epe_list)
        out_list = np.concatenate(out_list)

        epe = np.mean(epe_list)
        f1 = 100 * np.mean(out_list)

        return epe, f1
    else:
        return err_t_list, err_r_list, outliers, Time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, metavar='DIR',
                        default='/data/cky/KITTI/sequences',
                        help='path to dataset')
    parser.add_argument('--test_sequence', type=str, default='00')
    parser.add_argument('-cps', '--load_checkpoints', help="restore checkpoint")
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--starting_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=2, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--lr', '--learning_rate', default=4e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--max_r', type=float, default=10.)
    parser.add_argument('--max_t', type=float, default=2.)
    parser.add_argument('--use_reflectance', default=False)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--evaluate_interval', default=1, type=int, metavar='N',
                        help='Evaluate every \'evaluate interval\' epochs ')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.set_device(args.gpus[0])

    batch_size = args.batch_size

    model = torch.nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))
    if args.load_checkpoints is not None:
        model.load_state_dict(torch.load(args.load_checkpoints))
    model.to(device)

    bpnp = BPnP.apply

    def init_fn(x):
        return _init_fn(x, seed)

    dataset_test = DatasetVisibilityKittiSingle(args.data_path, max_r=args.max_r, max_t=args.max_t,
                                           split='test', use_reflectance=args.use_reflectance,
                                           test_sequence=args.test_sequence)
    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_test,
                                                shuffle=False,
                                                batch_size=1,
                                                num_workers=args.num_workers,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=True)
    if args.evaluate:
        with torch.no_grad():
            err_t_list, err_r_list, outliers, Time = test(args, TestImgLoader, model, bpnp, device, cal_pose=True)
            print(f"Mean trans error {np.mean(err_t_list):.5f}  Mean rotation error {np.mean(err_r_list):.5f}")
            print(f"Median trans error {np.median(err_t_list):.5f}  Median rotation error {np.median(err_r_list):.5f}")
            print(f"Outliers number {len(outliers)}/{len(TestImgLoader)}  Mean {Time / len(TestImgLoader):.5f} per frame")
        sys.exit()

    dataset_train = DatasetVisibilityKittiSingle(args.data_path, max_r=args.max_r, max_t=args.max_t,
                                           split='train', use_reflectance=args.use_reflectance,
                                           test_sequence=args.test_sequence)
    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 num_workers=args.num_workers,
                                                 worker_init_fn=init_fn,
                                                 collate_fn=merge_inputs,
                                                 drop_last=False,
                                                 pin_memory=True)
    print("Train length: ", len(TrainImgLoader))
    print("Test length: ", len(TestImgLoader))

    optimizer, scheduler = fetch_optimizer(args, len(TrainImgLoader), model)
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, SUM_FREQ=100)

    starting_epoch = args.starting_epoch
    min_val_err = 9999.
    for epoch in range(starting_epoch, args.epochs):
        # train
        train(args, epoch, TrainImgLoader, model, optimizer, scheduler, scaler, logger, device)

        if epoch % args.evaluate_interval == 0:
            epe, f1 = test(args, TestImgLoader, model, bpnp, device)
            print("Validation KITTI: %f, %f" % (epe, f1))

            results = {'kitti-epe': epe, 'kitti-f1': f1}
            logger.write_dict(results)

            torch.save(model.state_dict(), "./checkpoints/checkpoint.pth")

            if epe < min_val_err:
                min_val_err = epe
                torch.save(model.state_dict(), './checkpoints/best_model.pth')





