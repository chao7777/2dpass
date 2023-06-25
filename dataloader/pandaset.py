from torch.utils import data
import yaml
from pathlib import Path
import os
from torchvision import transforms as T
from PIL import Image
import random
import numpy as np
import pickle
import gzip
import json
from tools.pandaset.sensors import Intrinsics
from tools.pandaset import geometry
from .build import PIPELINE


class LoadPandasetInfo:
    def __init__(self, config, data_path, imageset='train'):
        self.config = config
        dataset_yaml = yaml.safe_load(open(config['dataset_params']['label_mapping']))
        if imageset == 'train':
            self.data_path = [os.path.join(data_path, x) for x in dataset_yaml['train_list']]
        else:
            self.data_path = [os.path.join(data_path, x) for x in dataset_yaml['val_list']]
        self.learning_map = dataset_yaml['learning_map']
        self.camera_list = ['back_camera', 'front_camera', 'front_left_camera', 'front_right_camera', 'left_camera', 'right_camera']
        self.get_available_scenes()
        print('total {} scenes'.format(str(len(self.ava_scenes))))
    
    def get_available_scenes(self):
        self.ava_scenes = []
        for data_dir in self.data_path:
            lidar_paths = os.listdir(os.path.join(data_dir, 'lidar'))
            for path in lidar_paths[:1]:
                if 'pkl.gz' not in path:continue
                self.ava_scenes.extend([data_dir, path.split('.pkl.gz')[0], x] for x in self.camera_list)
    
    def get_data_info(self, index):
        ava_scenes_idx = self.ava_scenes[index]
        lidar_data = pickle.load(gzip.open(os.path.join(ava_scenes_idx[0], 'lidar', ava_scenes_idx[1]+'.pkl.gz'))).values
        semseg_data = pickle.load(gzip.open(os.path.join(ava_scenes_idx[0], 'annotations', 'semseg', ava_scenes_idx[1]+'.pkl.gz'))).values
        cuboids_data = pickle.load(gzip.open(os.path.join(ava_scenes_idx[0], 'annotations', 'cuboids', ava_scenes_idx[1]+'.pkl.gz'))).values
        camera_data = Image.open(os.path.join(ava_scenes_idx[0], 'camera', ava_scenes_idx[2], ava_scenes_idx[1]+'.jpg'))
        camera_pose = json.load(open(os.path.join(ava_scenes_idx[0], 'camera', ava_scenes_idx[2], 'poses.json')))[int(ava_scenes_idx[1])]
        lidar_pose = json.load(open(os.path.join(ava_scenes_idx[0], 'lidar', 'poses.json')))[int(ava_scenes_idx[1])]
        camera_intrinsics = json.load(open(os.path.join(ava_scenes_idx[0], 'camera', ava_scenes_idx[2], 'intrinsics.json')))
        camera_intrinsics = Intrinsics(fx=camera_intrinsics['fx'],
                                        fy=camera_intrinsics['fy'],
                                        cx=camera_intrinsics['cx'],
                                        cy=camera_intrinsics['cy'])
        calib_info = {
            'camera_pose': camera_pose,
            'lidar_pose': lidar_pose,
            'camera_intrinsics': camera_intrinsics,
            'sub_dir': ava_scenes_idx[0],
            'idx': ava_scenes_idx[1],
            'camera_name': ava_scenes_idx[2],
        }
        data_dict = {
            'lidar_data': lidar_data,
            'cuboids_data': cuboids_data,
            'calib_info': calib_info,
            'semseg': semseg_data,
            'label_seg': np.vectorize(self.learning_map.__getitem__)(semseg_data),
            'camera_data': camera_data
        }
        if self.config['dataset_params']['use_seg_paste']:
            #更新data_dict
            pass
        projected_points2d, _, inner_indices = geometry.projection(lidar_points=data_dict['lidar_data'][:, :3], 
                                                                    camera_data=data_dict['camera_data'],
                                                                    camera_pose=camera_pose,
                                                                    camera_intrinsics=camera_intrinsics,
                                                                    filter_outliers=True)
        camera_lidar_data = data_dict['lidar_data'][inner_indices]
        ori_camera_semseg_data = data_dict['semseg'][inner_indices]
        camera_semseg_data = np.vectorize(self.learning_map.__getitem__)(ori_camera_semseg_data)
        data_dict.update({
            'xyz': camera_lidar_data[:, :3],
            'img': data_dict['camera_data'],
            'ori_semseg': camera_semseg_data,
            'labels': camera_semseg_data,
            'signal': camera_lidar_data[:, 3:4],
            'origin_len': camera_lidar_data.shape[0]
        })
        return data_dict

@PIPELINE.register_module()
class PandasetDateset(data.Dataset):
    def __init__(self, config, loader_config, data_path, imageset='train'):
        self.load_info = LoadPandasetInfo(config, data_path, imageset)
        self.config = config
        self.loader_config = loader_config
        self.max_volume_space = config['dataset_params']['max_volume_space']
        self.min_volume_space = config['dataset_params']['min_volume_space']
        self.imageset = imageset
        self.resize = config['dataset_params'].get('resize', False)
        self.trans_std = config['dataset_params']['trans_std']
        color_jitter = config['dataset_params']['color_jitter']
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
        self.image_normalizer = config['dataset_params'].get('image_normalizer', False)

    def __len__(self):
        return len(self.load_info.ava_scenes)

    def __getitem__(self, index):
        data_dict = self.load_info.get_data_info(index)
        lidar2ego_xyz = geometry.lidar_points_to_ego(data_dict['xyz'], data_dict['calib_info']['lidar_pose'])
        ref_pc = lidar2ego_xyz.copy()
        ref_labels = data_dict['labels'].copy() 
        ref_index = np.arange(len(ref_pc))

        mask_x = np.logical_and(lidar2ego_xyz[:, 0] > self.min_volume_space[0], lidar2ego_xyz[:, 0] < self.max_volume_space[0])
        mask_y = np.logical_and(lidar2ego_xyz[:, 1] > self.min_volume_space[1], lidar2ego_xyz[:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(lidar2ego_xyz[:, 2] > self.min_volume_space[2], lidar2ego_xyz[:, 2] < self.max_volume_space[2])
        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))
        lidar2ego_xyz = lidar2ego_xyz[mask]
        ref_pc = ref_pc[mask]
        data_dict['labels'] = data_dict['labels'][mask]
        ref_index = ref_index[mask]
        data_dict['signal'] = data_dict['signal'][mask]
        point_num = len(lidar2ego_xyz) 

        if self.loader_config['dropout_aug'] and self.imageset == 'train':
            dropout_ratio = np.random.random() * self.config['dataset_params']['max_dropout_ratio']
            drop_idx = np.where(np.random.random((lidar2ego_xyz.shape[0])) <= dropout_ratio)[0]

            if len(drop_idx) > 0:
                lidar2ego_xyz[drop_idx, :] = lidar2ego_xyz[0, :]
                data_dict['labels'][drop_idx, :] = data_dict['labels'][0, :]
                data_dict['signal'][drop_idx, :] = data_dict['signal'][0, :]
                ref_index[drop_idx] = ref_index[0]
        
        ego2lidar_xyz = geometry.ego_to_lidar_point(lidar2ego_xyz, data_dict['calib_info']['lidar_pose'])
        projected_points2d, _, inner_indices = geometry.projection(lidar_points=ego2lidar_xyz[:, :3], 
                                                                    camera_data=data_dict['img'],
                                                                    camera_pose=data_dict['calib_info']['camera_pose'],
                                                                    camera_intrinsics=data_dict['calib_info']['camera_intrinsics'],
                                                                    filter_outliers=True)
        points_img = np.ascontiguousarray(projected_points2d)
        if self.loader_config['rotate_aug']:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            lidar2ego_xyz[:, :2] = np.dot(lidar2ego_xyz[:, :2], j)
        
        if self.loader_config['flip_aug']:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                lidar2ego_xyz[:, 0] = -lidar2ego_xyz[:, 0]
            elif flip_type == 2:
                lidar2ego_xyz[:, 1] = -lidar2ego_xyz[:, 1]
            elif flip_type == 3:
                lidar2ego_xyz[:, :2] = -lidar2ego_xyz[:, :2]

        if self.loader_config['scale_aug']:
            noise_scale = np.random.uniform(0.95, 1.05)
            lidar2ego_xyz[:, 0] = noise_scale * lidar2ego_xyz[:, 0]
            lidar2ego_xyz[:, 1] = noise_scale * lidar2ego_xyz[:, 1]
        
        if self.loader_config['transform_aug']:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            lidar2ego_xyz[:, 0:3] += noise_translate
        
        point2img_index = np.arange(len(inner_indices))[inner_indices]
        feat = np.concatenate((lidar2ego_xyz, data_dict['signal']), axis=1)
        if self.resize:
            assert data_dict['img'].size[0] > self.resize[0]

            # scale image points
            points_img[:, 0] = float(self.resize[1]) / data_dict['img'].size[1] * np.floor(points_img[:, 0])
            points_img[:, 1] = float(self.resize[0]) / data_dict['img'].size[0] * np.floor(points_img[:, 1])

            # resize image
            data_dict['img'] = data_dict['img'].resize(self.resize, Image.BILINEAR)

        img_indices = points_img.astype(np.int64)

        if self.color_jitter is not None:
            data_dict['img'] = self.color_jitter(data_dict['img'])

        data_dict['img'] = np.array(data_dict['img'], dtype=np.float32, copy=False) / 255.

        if np.random.rand() < self.config['dataset_params']['flip2d']:
            data_dict['img'] = np.ascontiguousarray(data_dict['img'])
            img_indices[:, 1] = data_dict['img'].shape[1] - 1 - img_indices[:, 1]

        if self.image_normalizer:
            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            data_dict['img'] = (data_dict['img'] - mean) / std
        
        data_dict_res = {
            'point_feat': feat,
            'point_label':data_dict['labels'],
            'ref_xyz':ref_pc,
            'ref_label':ref_labels,
            'ref_index':ref_index,
            'mask':mask,
            'point_num':point_num,
            'origin_len':data_dict['origin_len'],
            'img':data_dict['img'],
            'img_indices':img_indices,
            'img_label':points_img,
            'point2img_index':point2img_index
        }
        return data_dict_res