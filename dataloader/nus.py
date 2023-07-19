import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T
from tools.nuscenes.utils import splits
from tools.nuscenes.nuscenes_ import NuScenes
import yaml
from pathlib import Path
import os
from PIL import Image
import mmcv
import numpy as np
from pyquaternion import Quaternion
from tools.nuscenes.utils.geometry_utils import view_points
from .build import PIPELINE


class LoadNuscenesInfo:
    def __init__(self, dataset_params, data_path, imageset='train', debug=False) -> None:
        if debug:
            version, scenes = 'v1.0-mini', splits.mini_train
        else:
            if imageset != 'test':
                version = 'v1.0-trainval'
                scenes = splits.train if imageset == 'train' else splits.val
            else:
                version, scenes = 'v1.0-test', splits.test

        self.split = imageset
        with open(dataset_params['label_mapping'], 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']
        self.data_path = data_path
        self.imageset = imageset
        self.img_view = ['CAM_FRONT', 'CAM_FRONT_RIGHT',
                         'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
        self.nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
        self.get_available_scenes()
        available_scene_names = [s['name'] for s in self.available_scenes]
        scenes = list(filter(lambda x: x in available_scene_names, scenes))
        scenes = set(
            [self.available_scenes[available_scene_names.index(s)]['token'] for s in scenes])
        self.get_path_infos_cam_lidar(scenes)
        print('Total %d scenes in the %s split' %
              (len(self.token_list), imageset))

    def loadImage(self, index, image_id):
        cam_sample_token = self.token_list[index]['cam_token'][image_id]
        cam = self.nusc.get('sample_data', cam_sample_token)
        image = Image.open(os.path.join(self.nusc.dataroot, cam['filename']))
        return image, cam_sample_token

    def loadDataByIndex(self, index):
        lidar_sample_token = self.token_list[index]['lidar_token']
        lidar_path = os.path.join(self.data_path, self.nusc.get(
            'sample_data', lidar_sample_token)['filename'])
        raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))

        if self.split == 'test':
            self.lidarseg_path = None
            annotated_data = np.expand_dims(
                np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            lidarseg_path = os.path.join(self.data_path, self.nusc.get(
                'lidarseg', lidar_sample_token)['filename'])
            annotated_data = np.fromfile(
                lidarseg_path, dtype=np.uint8).reshape((-1, 1))  # label

        pointcloud = raw_data[:, :4]
        sem_label = annotated_data
        inst_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
        return pointcloud, sem_label, inst_label, lidar_sample_token

    def get_available_scenes(self):
        # only for check if all the files are available
        self.available_scenes = []
        for scene in self.nusc.scene:
            scene_token = scene['token']
            scene_rec = self.nusc.get('scene', scene_token)
            sample_rec = self.nusc.get(
                'sample', scene_rec['first_sample_token'])
            sd_rec = self.nusc.get(
                'sample_data', sample_rec['data']['LIDAR_TOP'])
            has_more_frames = True
            scene_not_exist = False
            while has_more_frames:
                lidar_path, _, _ = self.nusc.get_sample_data(sd_rec['token'])
                if not Path(lidar_path).exists():
                    scene_not_exist = True
                    break
                else:
                    break

            if scene_not_exist:
                continue
            self.available_scenes.append(scene)

    def get_path_infos_cam_lidar(self, scenes):
        self.token_list = []

        for sample in self.nusc.sample:
            scene_token = sample['scene_token']
            lidar_token = sample['data']['LIDAR_TOP']  # 360 lidar

            if scene_token in scenes:
                cam_token = []
                for i in self.img_view:
                    cam_token.append(sample['data'][i])
                self.token_list.append({
                    'lidar_token': lidar_token,
                    'cam_token': cam_token
                })

    def get_cam_calib_info(self, index, image_id, lidar_sample_token):
        image, cam_sample_token = self.loadImage(index, image_id)

        cam_path, boxes_front_cam, cam_intrinsic = self.nusc.get_sample_data(
            cam_sample_token)
        pointsensor = self.nusc.get('sample_data', lidar_sample_token)
        cs_record_lidar = self.nusc.get(
            'calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pose_record_lidar = self.nusc.get(
            'ego_pose', pointsensor['ego_pose_token'])
        cam = self.nusc.get('sample_data', cam_sample_token)
        cs_record_cam = self.nusc.get(
            'calibrated_sensor', cam['calibrated_sensor_token'])
        pose_record_cam = self.nusc.get('ego_pose', cam['ego_pose_token'])

        calib_infos = {
            "lidar2ego_translation": cs_record_lidar['translation'],
            "lidar2ego_rotation": cs_record_lidar['rotation'],
            "ego2global_translation_lidar": pose_record_lidar['translation'],
            "ego2global_rotation_lidar": pose_record_lidar['rotation'],
            "ego2global_translation_cam": pose_record_cam['translation'],
            "ego2global_rotation_cam": pose_record_cam['rotation'],
            "cam2ego_translation": cs_record_cam['translation'],
            "cam2ego_rotation": cs_record_cam['rotation'],
            "cam_intrinsic": cam_intrinsic,
        }
        return calib_infos, cam_path

    def get_single_cam_data_info(self, index):
        pointcloud, sem_label, instance_label, lidar_sample_token = self.loadDataByIndex(
            index)
        sem_label = np.vectorize(self.learning_map.__getitem__)(sem_label)

        # get image feature
        image_id = np.random.randint(6)
        image, cam_sample_token = self.loadImage(index, image_id)
        calib_infos, cam_path = self.get_cam_calib_info(
            index, image_id, cam_sample_token)

        data_dict = {
            'xyz': pointcloud[:, :3],
            'img': image,
            'calib_infos': calib_infos,
            'labels': sem_label.astype(np.uint8),
            'signal': pointcloud[:, 3:4],
            'origin_len': len(pointcloud),
            'cam_path': cam_path
        }
        return data_dict, lidar_sample_token

    def get_multy_cam_data_info(self, index):
        pointcloud, sem_label, instance_label, lidar_sample_token = self.loadDataByIndex(
            index)
        sem_label = np.vectorize(self.learning_map.__getitem__)(sem_label)
        camera_info_dict = {}
        for image_id in range(6):
            camera_name = self.img_view[image_id]
            image, cam_sample_token = self.loadImage(index, image_id)
            calib_infos, cam_path = self.get_cam_calib_info(
                index, image_id, cam_sample_token)
            camera_info_dict.update({camera_name: {}})
            camera_info_dict[camera_name].update({
                'img': image,
                'cam_path': cam_path,
                'calib_infos': calib_infos
            })
        data_dict = {
            'xyz': pointcloud[:, :3],
            'labels': sem_label.astype(np.uint8),
            'signal': pointcloud[:, 3:4],
            'origin_len': len(pointcloud),
            'camera_info_dict': camera_info_dict
        }
        return data_dict, lidar_sample_token


@PIPELINE.register_module()
class NuscenseDateset(Dataset):
    def __init__(self, imageset, data_path, dataset_params, dataload_config):
        super().__init__()
        self.debug = dataload_config['debug']
        self.imageset = imageset
        self.dataset_params = dataset_params
        self.loader_config = dataload_config
        self.load_info = LoadNuscenesInfo(
            dataset_params, data_path, imageset, self.debug)
        self.resize = dataset_params.get('resize', False)
        color_jitter = dataset_params['color_jitter']
        self.color_jitter = T.ColorJitter(
            *color_jitter) if color_jitter else None
        self.image_normalizer = dataset_params.get('image_normalizer', False)
        self.trans_std = dataset_params['trans_std']
        self.max_volume_space = dataset_params['max_volume_space']
        self.min_volume_space = dataset_params['min_volume_space']

    def __len__(self):
        return 100 if self.debug else len(self.load_info.token_list)

    def __getitem__(self, index):
        'Generates one sample of data'
        data, root = self.load_info.get_single_cam_data_info(index)
        ref_pc = data['xyz'].copy()
        ref_labels = data['labels'].copy()
        ref_index = np.arange(len(ref_pc))

        data, mask = self.filter_min_max_pts(data)
        ref_pc = ref_pc[mask]
        ref_index = ref_index[mask]
        point_num = len(data['xyz'])

        # dropout points
        data['xyz'], data['labels'], data['signal'], ref_index = self.dropout_aug(
            data['xyz'], data['labels'], data['signal'], ref_index)

        keep_idx, _, points_img, depths = self.map_pointcloud_to_image(
            data['xyz'], (data['img'].size[1], data['img'].size[0]), data['calib_infos'])
        points_img = np.ascontiguousarray(np.fliplr(points_img))
        # self.vis(points_img, labels, keep_idx, data['cam_path'])

        data['xyz'] = self.lidar_aug(data['xyz'])

        points_img = points_img[keep_idx]
        img_label = data['labels'][keep_idx]
        point2img_index = np.arange(len(keep_idx))[keep_idx]
        feat = np.concatenate((data['xyz'], data['signal']), axis=1)

        ### 2D Augmentation ###
        data['img'], points_img = self.resize_2d(data['img'], points_img)
        img_indices = points_img.astype(np.int64)

        # 2D augmentation
        data['img'] = self.color_jitter_2d(data['img'])
        # 2D augmentation
        data['img'], img_indices = self.flip_2d(data['img'], img_indices)
        # normalize image
        data['img'] = self.normalize_2d(data['img'])

        data_dict = {
            'point_feat': torch.tensor(feat),
            'point_label': torch.tensor(data['labels']),
            'ref_xyz': ref_pc,
            'ref_label': ref_labels,
            'ref_index': ref_index,
            'mask': mask,
            'point_num': point_num,
            'origin_len': data['origin_len'],
            'root': root,
            'img': torch.tensor(data['img']),
            'img_indices': img_indices,
            'img_label': torch.tensor(img_label),
            'point2img_index': torch.tensor(point2img_index).long()
        }

        return data_dict

    def lidar_aug(self, xyz):
        # random data augmentation by rotation
        xyz = self.rotate_aug(xyz)

        # random data augmentation by flip x , y or x+y
        xyz = self.flip_aug(xyz)
        xyz = self.scale_aug(xyz)
        xyz = self.transform_aug(xyz)
        return xyz

    def color_jitter_2d(self, img):
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        return img

    def normalize_2d(self, image):
        image = np.array(image, dtype=np.float32, copy=False) / 255.
        if self.image_normalizer:
            mean, std = self.image_normalizer['mean'], self.image_normalizer['std']
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std
        return image

    def flip_2d(self, image, img_indices):
        if np.random.rand() < self.loader_config['flip2d']:
            image = np.ascontiguousarray(np.fliplr(image))
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]
        return image, img_indices

    def resize_2d(self, image, points_img):
        if self.resize:
            assert image.size[0] > self.resize[0]

            # scale image points
            points_img[:, 0] = float(self.resize[1]) / \
                image.size[1] * np.floor(points_img[:, 0])
            points_img[:, 1] = float(self.resize[0]) / \
                image.size[0] * np.floor(points_img[:, 1])

            # resize image
            image = image.resize(self.resize, Image.BILINEAR)
        return image, points_img

    def transform_aug(self, xyz):
        if self.loader_config['transform_aug']:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(
                                            0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate
        return xyz

    def scale_aug(self, xyz):
        if self.loader_config['scale_aug']:
            noise_scale = np.random.uniform(*self.loader_config['scale_aug'])
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
        return xyz

    def flip_aug(self, xyz):
        if self.loader_config['flip_aug']:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        return xyz

    def rotate_aug(self, xyz):
        if self.loader_config['rotate_aug']:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)
        return xyz

    def filter_min_max_pts(self, data_info):
        mask_x = np.logical_and(data_info['xyz'][:, 0] > self.min_volume_space[0],
                                data_info['xyz'][:, 0] < self.max_volume_space[0])
        mask_y = np.logical_and(data_info['xyz'][:, 1] > self.min_volume_space[1],
                                data_info['xyz'][:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(data_info['xyz'][:, 2] > self.min_volume_space[2],
                                data_info['xyz'][:, 2] < self.max_volume_space[2])
        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))
        data_info['xyz'] = data_info['xyz'][mask]
        data_info['labels'] = data_info['labels'][mask]
        data_info['signal'] = data_info['signal'][mask]
        return data_info, mask

    def dropout_aug(self, xyz, labels, sig, ref_index):
        if self.loader_config['dropout_aug'] and self.imageset == 'train':
            dropout_ratio = np.random.random(
            ) * self.dataset_params['max_dropout_ratio']
            drop_idx = np.where(np.random.random(
                (xyz.shape[0])) <= dropout_ratio)[0]

            if len(drop_idx) > 0:
                xyz[drop_idx, :] = xyz[0, :]
                labels[drop_idx, :] = labels[0, :]
                sig[drop_idx, :] = sig[0, :]
                ref_index[drop_idx] = ref_index[0]
        return xyz, labels, sig, ref_index

    def vis(self, points_img, labels, keep_idx, cam_path):
        # 可视化某一分割类别的数据集
        import cv2
        img = cv2.imread(cam_path)
        k_img = points_img[keep_idx]
        label = labels[keep_idx]
        veg = np.ones(label.shape[0], dtype=bool)
        veg = np.logical_and(veg, label[:, 0] == 16)
        veg_img = k_img[veg]
        for i in range(len(veg_img)):
            cv2.circle(img, (int(veg_img[i][1]), int(
                veg_img[i][0])), 1, (255, 0, 0), 4)
        cv2.imwrite('work_dirs/veg.jpg', img)

    def map_pointcloud_to_image(self, pc, im_shape, info):
        """
        Maps the lidar point cloud to the image.
        :param pc: (3, N)
        :param im_shape: image to check size and debug
        :param info: dict with calibration infos
        :param im: image, only for visualization
        :return:
        """
        pc = pc.copy().T

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
        pc = Quaternion(info['lidar2ego_rotation']).rotation_matrix @ pc
        pc = pc + np.array(info['lidar2ego_translation'])[:, np.newaxis]

        # Second step: transform to the global frame.
        pc = Quaternion(info['ego2global_rotation_lidar']).rotation_matrix @ pc
        pc = pc + np.array(info['ego2global_translation_lidar'])[:, np.newaxis]

        # Third step: transform into the ego vehicle frame for the timestamp of the image.
        pc = pc - np.array(info['ego2global_translation_cam'])[:, np.newaxis]
        pc = Quaternion(info['ego2global_rotation_cam']).rotation_matrix.T @ pc

        # Fourth step: transform into the camera.
        pc = pc - np.array(info['cam2ego_translation'])[:, np.newaxis]
        pc = Quaternion(info['cam2ego_rotation']).rotation_matrix.T @ pc

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc[2, :]

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(pc, np.array(
            info['cam_intrinsic']), normalize=True)

        # Cast to float32 to prevent later rounding errors
        points = points.astype(np.float32)

        # Remove points that are either outside or behind the camera.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0)
        mask = np.logical_and(mask, points[0, :] > 0)
        mask = np.logical_and(mask, points[0, :] < im_shape[1])
        mask = np.logical_and(mask, points[1, :] > 0)
        mask = np.logical_and(mask, points[1, :] < im_shape[0])

        return mask, pc.T, points.T[:, :2], depths


@PIPELINE.register_module()
class NuscenseDatesetDepth(NuscenseDateset):
    def __init__(self, imageset, data_path, dataset_params, dataload_config):
        super().__init__(imageset, data_path, dataset_params, dataload_config)

    def __len__(self):
        return 100 if self.debug else len(self.load_info.token_list)

    def __getitem__(self, index):
        data, _ = self.load_info.get_multy_cam_data_info(index)
        data, _ = self.filter_min_max_pts(data)
        ref_labels = data['labels'].copy()
        ref_index = np.arange(len(data['xyz']))

        point_num = len(data['xyz'])
        # dropout points
        data['xyz'], data['labels'], data['signal'], ref_index = self.dropout_aug(
            data['xyz'], data['labels'], data['signal'], ref_index)
        img_lables, img_indices, imgs, point2img_indexs, lidar_depths = [], [], [], [], [], []
        ida_mats, intrin_mats, sensor2ego_mats = [], [], [], []
        for camera_info in data['camera_info_dict']:
            w, h = camera_info['img'].size[:2]
            keep_idx, _, points_img, depths = self.map_pointcloud_to_image(
                data['xyz'], (h, w), camera_info['calib_infos'])
            lidar_depth = np.concatenate(
                [points_img[keep_idx][:2, :].T, depths[keep_idx][:, None]], axis=1).astype(np.float32)
            points_img = np.ascontiguousarray(np.fliplr(points_img[keep_idx]))
            img_lables.append(torch.tensor(data['labels'][keep_idx]))
            point2img_indexs.append(torch.tensor(
                np.arange(len(keep_idx))[keep_idx]).long)

            resize, img_scale, flip, rotate_ida = self.sample_ida_augmentation(
                w)
            point_depth_augmented = self.depth_transform(
                lidar_depth, resize, img_scale, flip, rotate_ida)
            lidar_depths.append(point_depth_augmented)
            camera_info['img'] = self.color_jitter_2d(camera_info['img'])
            img, ida_mat, img_indice = self.img_transform(
                camera_info['img'], resize, img_scale, flip, rotate_ida, points_img, (w, h))
            # img = self.normalize_2d(img)
            img = mmcv.imnormalize(np.array(
                img), self.loader_config['ida_aug_conf']['img_mean'], self.loader_config['ida_aug_conf']['img_std'], True)
            imgs.append(torch.from_numpy(img).permute(2, 0, 1))
            ida_mats.append(ida_mat)
            img_indices.append(torch.tensor(img_indice))
            intrin_mats.append(self.get_intrin_mat(camera_info['calib_info']))
            sensor2ego_mats.append(
                self.get_sensor2ego_mat(camera_info['calib_info']))
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()
        bda_rot = self.bev_transform(rotate_bda, scale_bda, flip_dx, flip_dy)
        bda_mat = self.get_bda_mat(bda_rot)
        data['xyz'] = self.lidar_aug(data['xyz'])
        point = torch.tensor(np.concatenate(
            data['xyz'][keep_idx], data['signal'][keep_idx], axis=1))

        data_dict = {
            'point_feat': point,
            'point_label': torch.tensor(data['labels']),
            'point_num': point_num,
            'ref_labels': ref_labels,
            'ref_index': ref_index,
            'origin_len': data['origin_len'],
            'img': torch.stack(imgs),
            'img_label': torch.stack(img_lables),
            'img_indices': torch.stack(img_indices),
            'point2img_indexs': torch.stack(point2img_indexs),
            'ida_mats': torch.stack(ida_mats),
            'sensor2ego_mats': torch.stack(sensor2ego_mats),
            'intrin_mats': torch.stack(intrin_mats),
            'bda_mat': bda_mat,
            'lidar_depths': torch.stack(lidar_depths)
        }
        return data_dict

    def get_bda_mat(self, bda_rot):
        bda_mat = torch.zeros((4, 4))
        bda_mat[3, 3] = 1
        bda_mat[:3, :3] = bda_rot
        return bda_mat

    def bev_transform(self, rotate_angle, scale_ratio, flip_dx, flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor(
                [[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor(
                [[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        return rot_mat

    def get_sensor2ego_mat(self, calib_info):
        w, x, y, z = calib_info['cam2ego_rotation']
        # cur ego to sensor
        keysensor2keyego = torch.zeros((4, 4))
        keysensor2keyego[3, 3] = 1
        keysensor2keyego[:3, :3] = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keysensor2keyego[:3, -
                         1] = torch.Tensor(calib_info['cam2ego_translation'])
        # global sensor to cur ego
        w, x, y, z = calib_info['ego2global_rotation_cam']
        keyego2global = torch.zeros((4, 4))
        keyego2global[3, 3] = 1
        keyego2global[:3, :3] = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keyego2global[:3, -
                      1] = torch.Tensor(calib_info['ego2global_translation_cam'])
        global2keyego = keyego2global.inverse()
        keyego2keysensor = keysensor2keyego.inverse()
        sweepsensor2keyego = global2keyego @ keyego2global @ keyego2keysensor
        return sweepsensor2keyego

    def get_intrin_mat(self, calib_info):
        intrin_mat = torch.zeros((4, 4))
        intrin_mat[3, 3] = 1
        intrin_mat[:3, :3] = torch.tensor(calib_info['cam_intrinsic'])
        return intrin_mat

    def sample_ida_augmentation(self, w):
        """Generate ida augmentation values based on ida_config."""
        img_scale = self.loader_config['ida_aug_conf']['img_scale']
        resize = img_scale[1] / w
        flip = False
        if self.loader_config['ida_aug_conf']['rand_flip'] and np.random.choice([0, 1]):
            flip = True
        rotate_ida = np.random.uniform(
            *self.loader_config['ida_aug_conf']['rot_lim'])
        return resize, img_scale, flip, rotate_ida

    def sample_bda_augmentation(self):
        rotate_bda = np.random.uniform(
            *self.loader_config['bda_aug_conf']['rot_lim'])
        scale_bda = np.random.uniform(
            *self.loader_config['bda_aug_conf']['scale_lim'])
        flip_dx = np.random.uniform(
        ) < self.loader_config['bda_aug_conf']['flip_dx_ratio']
        flip_dy = np.random.uniform(
        ) < self.loader_config['bda_aug_conf']['flip_dy_ratio']
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def depth_transform(self, cam_depth, resize, img_scale, flip, rotate):
        """Transform depth based on ida augmentation configuration.

        Args:
            cam_depth (np array): Nx3, 3: x,y,d.
            resize (float): Resize factor.
            resize_dims (list): Final dimension.
            flip (bool): Whether to flip.
            rotate (float): Rotation value.

        Returns:
            np array: [h/down_ratio, w/down_ratio, d]
        """
        H, W = img_scale
        cam_depth[:, :2] = cam_depth[:, :2] * resize
        if flip:
            cam_depth[:, 0] = img_scale[1] - cam_depth[:, 0]

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

        depth_map = np.zeros(img_scale)
        valid_mask = ((depth_coords[:, 1] < img_scale[0])
                      & (depth_coords[:, 0] < img_scale[1])
                      & (depth_coords[:, 1] >= 0)
                      & (depth_coords[:, 0] >= 0))
        depth_map[depth_coords[valid_mask, 1],
                  depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

        return torch.Tensor(depth_map)

    def img_transform(self, img, resize, img_scale, flip, rotate, points_img, img_size):
        ida_rot, ida_tran = torch.eye(2), torch.zeros(2)
        img = img.resize((img_scale[1], img_scale[0]), Image.BILINEAR)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        points_img[:, 0] = float(img_scale[1]) / \
            img_size[1] * np.floor(points_img[:, 0])
        points_img[:, 1] = float(img_scale[0]) / \
            img_size[0] * np.floor(points_img[:, 1])
        img_indice = points_img.astype(np.int64)
        if flip:
            img_indice[:, 1] = img_scale[0] - 1 - img_indice[:, 1]
        # post-homography transformation
        ida_rot *= resize
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([img_scale[1], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        h = rotate / 180 * np.pi
        A = torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])
        b = torch.Tensor([img_scale[1], img_scale[0]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = ida_rot.new_zeros(4, 4)
        ida_mat[3, 3] = 1
        ida_mat[2, 2] = 1
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 3] = ida_tran
        return img, ida_mat, img_indice
