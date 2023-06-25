import torch
from torch.utils import data
from torchvision import transforms as T
from tools.nuscenes.utils import splits
from tools.nuscenes.nuscenes_ import NuScenes
import yaml
from pathlib import Path
import os
from PIL import Image
import numpy as np
from pyquaternion import Quaternion
from tools.nuscenes.utils.geometry_utils import view_points
from .build import PIPELINE

class LoadNuscenesInfo:
    def __init__(self, config, data_path, imageset='train') -> None:
        if config.debug:
            version, scenes = 'v1.0-mini', splits.mini_train
        else:
            if imageset != 'test':
                version = 'v1.0-trainval'
                scenes = splits.train if imageset == 'train' else splits.val
            else:
                version, scenes = 'v1.0-test', splits.test

        self.split = imageset
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']
        self.data_path = data_path
        self.imageset = imageset
        self.img_view = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
        self.nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
        self.get_available_scenes()
        available_scene_names = [s['name'] for s in self.available_scenes]
        scenes = list(filter(lambda x: x in available_scene_names, scenes))
        scenes = set([self.available_scenes[available_scene_names.index(s)]['token'] for s in scenes])
        self.get_path_infos_cam_lidar(scenes)
        print('Total %d scenes in the %s split' % (len(self.token_list), imageset))
    
    def loadImage(self, index, image_id):
        cam_sample_token = self.token_list[index]['cam_token'][image_id]
        cam = self.nusc.get('sample_data', cam_sample_token)
        image = Image.open(os.path.join(self.nusc.dataroot, cam['filename']))
        return image, cam_sample_token

    def loadDataByIndex(self, index):
        lidar_sample_token = self.token_list[index]['lidar_token']
        lidar_path = os.path.join(self.data_path, self.nusc.get('sample_data', lidar_sample_token)['filename'])
        raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))

        if self.split == 'test':
            self.lidarseg_path = None
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            lidarseg_path = os.path.join(self.data_path, self.nusc.get('lidarseg', lidar_sample_token)['filename'])
            annotated_data = np.fromfile(lidarseg_path, dtype=np.uint8).reshape((-1, 1))  # label

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
            sample_rec = self.nusc.get('sample', scene_rec['first_sample_token'])
            sd_rec = self.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
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
    
    def get_data_info(self, index):
        pointcloud, sem_label, instance_label, lidar_sample_token = self.loadDataByIndex(index)
        sem_label = np.vectorize(self.learning_map.__getitem__)(sem_label)

        # get image feature
        image_id = np.random.randint(6)
        image, cam_sample_token = self.loadImage(index, image_id)

        cam_path, boxes_front_cam, cam_intrinsic = self.nusc.get_sample_data(cam_sample_token)
        pointsensor = self.nusc.get('sample_data', lidar_sample_token)
        cs_record_lidar = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pose_record_lidar = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        cam = self.nusc.get('sample_data', cam_sample_token)
        cs_record_cam = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
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

        data_dict = {}
        data_dict['xyz'] = pointcloud[:, :3]
        data_dict['img'] = image
        data_dict['calib_infos'] = calib_infos
        data_dict['labels'] = sem_label.astype(np.uint8)
        data_dict['signal'] = pointcloud[:, 3:4]
        data_dict['origin_len'] = len(pointcloud)
        data_dict['cam_path'] = cam_path

        return data_dict, lidar_sample_token
    
@PIPELINE.register_module()
class NuscenseDateset(data.Dataset):
    def __init__(self, config, loader_config, data_path, imageset='train'):
        self.debug = config['debug']
        self.imageset = imageset
        self.config = config
        self.loader_config = loader_config
        self.load_info = LoadNuscenesInfo(config, data_path, imageset)
        self.resize = config['dataset_params'].get('resize', False)
        color_jitter = config['dataset_params']['color_jitter']
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
        self.image_normalizer = config['dataset_params'].get('image_normalizer', False)
        self.trans_std = config['dataset_params']['trans_std']
        self.max_volume_space = config['dataset_params']['max_volume_space']
        self.min_volume_space = config['dataset_params']['min_volume_space']
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


    def __len__(self):
        return 100 if self.debug else len(self.load_info.token_list)

    def __getitem__(self, index):
        'Generates one sample of data'
        data, root = self.load_info.get_data_info(index)

        xyz = data['xyz']
        labels = data['labels']
        sig = data['signal']
        origin_len = data['origin_len']

        # load 2D data
        image = data['img']
        calib_infos = data['calib_infos']

        ref_pc = xyz.copy()
        ref_labels = labels.copy()
        ref_index = np.arange(len(ref_pc))

        mask_x = np.logical_and(xyz[:, 0] > self.min_volume_space[0], xyz[:, 0] < self.max_volume_space[0])
        mask_y = np.logical_and(xyz[:, 1] > self.min_volume_space[1], xyz[:, 1] < self.max_volume_space[1])
        mask_z = np.logical_and(xyz[:, 2] > self.min_volume_space[2], xyz[:, 2] < self.max_volume_space[2])
        mask = np.logical_and(mask_x, np.logical_and(mask_y, mask_z))

        xyz = xyz[mask]
        ref_pc = ref_pc[mask]
        labels = labels[mask]
        ref_index = ref_index[mask]
        sig = sig[mask]
        point_num = len(xyz)

        # dropout points
        if self.loader_config['dropout_aug'] and self.imageset == 'train':
            dropout_ratio = np.random.random() * self.config['dataset_params']['max_dropout_ratio']
            drop_idx = np.where(np.random.random((xyz.shape[0])) <= dropout_ratio)[0]

            if len(drop_idx) > 0:
                xyz[drop_idx, :] = xyz[0, :]
                labels[drop_idx, :] = labels[0, :]
                sig[drop_idx, :] = sig[0, :]
                ref_index[drop_idx] = ref_index[0]

        keep_idx, _, points_img, depths = self.map_pointcloud_to_image(xyz, (image.size[1], image.size[0]), calib_infos)
        # lidar_depths = np.concatenate([points_img[keep_idx][:2, :].T, depths[keep_idx][:, None]], axis=1).astype(np.float32)
        # if self.is_ida_aug:
        #     resize, _, crop, flip, rotate_ida = self.sample_ida_augmentation()
        #     point_depth_augmented = self.depth_transform(lidar_depths, resize, self.ida_aug_conf['final_dim'], crop, flip, rotate_ida)
        #     depth_label = torch.stack([point_depth_augmented])
        points_img = np.ascontiguousarray(np.fliplr(points_img))
        
        # 可视化某一分割类别的数据集
        """
        import cv2
        img = cv2.imread(data['cam_path'])
        k_img = points_img[keep_idx]
        label = labels[keep_idx]
        veg = np.ones(label.shape[0], dtype=bool)
        veg = np.logical_and(veg, label[:,0]==16)
        veg_img = k_img[veg]
        for i in range(len(veg_img)):
            cv2.circle(img, (int(veg_img[i][1]), int(veg_img[i][0])), 1, (255,0,0), 4)
        cv2.imwrite('work_dirs/veg.jpg', img)
        """
        
        # random data augmentation by rotation
        if self.loader_config['rotate_aug']:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.loader_config['flip_aug']:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        if self.loader_config['scale_aug']:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]

        if self.loader_config['transform_aug']:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        points_img = points_img[keep_idx]
        img_label = labels[keep_idx]
        point2img_index = np.arange(len(keep_idx))[keep_idx]
        feat = np.concatenate((xyz, sig), axis=1)

        ### 2D Augmentation ###
        if self.resize:
            assert image.size[0] > self.resize[0]

            # scale image points
            points_img[:, 0] = float(self.resize[1]) / image.size[1] * np.floor(points_img[:, 0])
            points_img[:, 1] = float(self.resize[0]) / image.size[0] * np.floor(points_img[:, 1])

            # resize image
            image = image.resize(self.resize, Image.BILINEAR)

        img_indices = points_img.astype(np.int64)

        # 2D augmentation
        if self.color_jitter is not None:
            image = self.color_jitter(image)

        image = np.array(image, dtype=np.float32, copy=False) / 255.

        # 2D augmentation
        if np.random.rand() < self.config['dataset_params']['flip2d']:
            image = np.ascontiguousarray(np.fliplr(image))
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]

        # normalize image
        if self.image_normalizer:
            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std

        data_dict = {}
        data_dict['point_feat'] = feat
        data_dict['point_label'] = labels
        data_dict['ref_xyz'] = ref_pc
        data_dict['ref_label'] = ref_labels
        data_dict['ref_index'] = ref_index
        data_dict['mask'] = mask
        data_dict['point_num'] = point_num
        data_dict['origin_len'] = origin_len
        data_dict['root'] = root

        data_dict['img'] = image
        data_dict['img_indices'] = img_indices
        data_dict['img_label'] = img_label
        data_dict['point2img_index'] = point2img_index
        
        return data_dict
    
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
        points = view_points(pc, np.array(info['cam_intrinsic']), normalize=True)

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
