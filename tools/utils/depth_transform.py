from textwrap import indent
import mmcv
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from tools.nuscenes.utils.data_classes import Box, LidarPointCloud
from tools.nuscenes.utils.geometry_utils import view_points
import os
import random
import cv2

class NuscenesDepth:
    def __init__(self, data_root, info_path):
        self.info = mmcv.load(info_path)
        self.data_root = data_root
        self.ida_aug_conf = {
            'resize_lim': (0.386, 0.55),
            'final_dim': (256, 704),
            'rot_lim': (-5.4, 5.4),
            'H': 900,
            'W': 1600,
            'rand_flip': True,
            'bot_pct_lim': (0.0, 0.0),
            'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
            'Ncams': 6
        }
    
    def get_depth_label(self):
        cam_infos, lidar_infos = [], []
        info = self.info[0]
        # cams = np.random.choice(self.ida_aug_conf['cams'], self.ida_aug_conf['Ncams'], replace=False)
        cams = self.ida_aug_conf['cams']
        cam_infos.append(info['cam_infos'])
        lidar_infos.append(info['lidar_infos'])
        assert len(cam_infos) > 0
        sweep_lidar_depth, sweep_lidar_points = [], []
        for lidar_info in lidar_infos:
            lidar_path = lidar_info['LIDAR_TOP']['filename']
            lidar_points = np.fromfile(os.path.join(self.data_root, lidar_path), dtype=np.float32, count=-1).reshape(-1, 5)[..., :4]
            sweep_lidar_points.append(lidar_points)
        for cam in cams:
            lidar_depth = []
            resize, _, crop, flip, rotate_ida = self.sample_ida_augmentation()

            for sweep_idx, cam_info in enumerate(cam_infos):
                img = Image.open(os.path.join(self.data_root, cam_info[cam]['filename']))
                if sweep_idx == 0:
                    point_depth = self.get_lidar_depth(sweep_lidar_points[sweep_idx], img, lidar_infos[sweep_idx], cam_info[cam])
                    point_depth_augmented = self.depth_transform(point_depth, resize, self.ida_aug_conf['final_dim'], crop, flip, rotate_ida)
                    lidar_depth.append(point_depth_augmented)

            sweep_lidar_depth.append(torch.stack(lidar_depth))
        depth_label = torch.stack(sweep_lidar_depth).permute(1, 0, 2, 3)
    
    def map_pointcloud_to_image(self, lidar_points, img, lidar_calibrated_sensor, lidar_ego_pose, cam_calibrated_sensor, cam_ego_pose, min_dist=0.0):

        # Points live in the point sensor frame. So they need to be
        # transformed via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle
        # frame for the timestamp of the sweep.

        lidar_points = LidarPointCloud(lidar_points.T)
        lidar_points.rotate(Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
        lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))

        # Second step: transform from ego to the global frame.
        lidar_points.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
        lidar_points.translate(np.array(lidar_ego_pose['translation']))

        # Third step: transform from global into the ego vehicle
        # frame for the timestamp of the image.
        lidar_points.translate(-np.array(cam_ego_pose['translation']))
        lidar_points.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        lidar_points.translate(-np.array(cam_calibrated_sensor['translation']))
        lidar_points.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = lidar_points.points[2, :]
        coloring = depths

        # Take the actual picture (matrix multiplication with camera-matrix
        # + renormalization).
        points = view_points(lidar_points.points[:3, :],
                            np.array(cam_calibrated_sensor['camera_intrinsic']),
                            normalize=True)

        # Remove points that are either outside or behind the camera.
        # Leave a margin of 1 pixel for aesthetic reasons. Also make
        # sure points are at least 1m in front of the camera to avoid
        # seeing the lidar points on the camera casing for non-keyframes
        # which are slightly out of sync.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 0)
        mask = np.logical_and(mask, points[0, :] < img.size[0])
        mask = np.logical_and(mask, points[1, :] > 0)
        mask = np.logical_and(mask, points[1, :] < img.size[1])
        points = points[:, mask]
        coloring = coloring[mask]

        return points, coloring

    def depth_transform(self, cam_depth, resize, resize_dims, crop, flip, rotate):
        """Transform depth based on ida augmentation configuration.

        Args:
            cam_depth (np array): Nx3, 3: x,y,d.
            resize (float): Resize factor.
            resize_dims (list): Final dimension.
            crop (list): x1, y1, x2, y2
            flip (bool): Whether to flip.
            rotate (float): Rotation value.

        Returns:
            np array: [h/down_ratio, w/down_ratio, d]
        """

        H, W = resize_dims
        cam_depth[:, :2] = cam_depth[:, :2] * resize
        cam_depth[:, 0] -= crop[0]
        cam_depth[:, 1] -= crop[1]
        if flip:
            cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

        cam_depth[:, 0] -= W / 2.0
        cam_depth[:, 1] -= H / 2.0

        h = rotate / 180 * np.pi
        rot_matrix = [
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ]
        cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

        cam_depth[:, 0] += W / 2.0
        cam_depth[:, 1] += H / 2.0

        depth_coords = cam_depth[:, :2].astype(np.int16)

        depth_map = np.zeros(resize_dims)
        valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                    & (depth_coords[:, 0] < resize_dims[1])
                    & (depth_coords[:, 1] >= 0)
                    & (depth_coords[:, 0] >= 0))
        depth_map[depth_coords[valid_mask, 1], depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

        return torch.Tensor(depth_map)


    def sample_ida_augmentation(self):
        """Generate ida augmentation values based on ida_config."""
        H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']
        resize = np.random.uniform(*self.ida_aug_conf['resize_lim'])
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.random.uniform(*self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
        crop_w = int(np.random.uniform(0, max(0, newW - fW)))
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
            flip = True
        rotate_ida = np.random.uniform(*self.ida_aug_conf['rot_lim'])
        return resize, resize_dims, crop, flip, rotate_ida
    
    def get_lidar_depth(self, lidar_points, img, lidar_info, cam_info):
        lidar_calibrated_sensor = lidar_info['LIDAR_TOP']['calibrated_sensor']
        lidar_ego_pose = lidar_info['LIDAR_TOP']['ego_pose']
        cam_calibrated_sensor = cam_info['calibrated_sensor']
        cam_ego_pose = cam_info['ego_pose']
        pts_img, depth = self.map_pointcloud_to_image(
            lidar_points.copy(), img, lidar_calibrated_sensor.copy(),
            lidar_ego_pose.copy(), cam_calibrated_sensor, cam_ego_pose)
        pts_img = pts_img[:2, :].T
        ori_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for pts in pts_img:
            cv2.circle(ori_img, (int(pts[0]), int(pts[1])), 1, (255,0,0), 4)
        cv2.imwrite('/home/qi.chao/c7/mycode/2dpass/work_dirs/nuscenes/bev_depth/jpg/test.jpg', ori_img)
        return np.concatenate([pts_img[:2, :].T, depth[:, None]], axis=1).astype(np.float32)


def main():
    data_root = '/share-global/xinli.xu/dataset/nuscenes'
    info_json_path = 'work_dirs/nuscenes/bev_depth/info/nuscenes_infos_train_test.json'
    nuscense_depth = NuscenesDepth(data_root, info_json_path)
    nuscense_depth.get_depth_label()

if __name__ == '__main__':
    main()

