import pickle
import gzip
import numpy as np
import cv2
from pyquaternion import Quaternion
from tqdm import tqdm
import yaml
from pathlib import Path
import json
from get_seed import MultiProcess
import os
from PIL import Image
from tools.nuscenes.utils import splits
from tools.nuscenes.utils.geometry_utils import view_points
from tools.nuscenes.nuscenes_ import NuScenes
from tools.pandaset import geometry
from tools.pandaset.sensors import Intrinsics
import random


class PandasetCount:
    def __init__(self, config_file, data_dir, save_jpg_dir):
        yaml_conf = yaml.load(open(config_file), Loader=yaml.FullLoader)
        self.train_list, self.val_list = yaml_conf['train_list'], yaml_conf['val_list']
        self.labels = yaml_conf['labels']
        self.count_dict = {
            'semseg_count': {},
            'cuboids_count': {}
        }
        self.data_dir = data_dir
        self.save_jpg_dir = save_jpg_dir
    
    def count(self):
        p_list = self.train_list[:1]
        process_num = min(len(p_list), 10)
        MultiProcess.multi_process_entr(
            MultiProcess, p_list, process_num, self.save_jpg_dir, self.task_func
        )
        json.dump(self.count_dict['semseg_count'], open(os.path.join(self.save_jpg_dir, 'json', 'semseg_count.json'), 'w'), indent=4)
        json.dump(self.count_dict['cuboids_count'], open(os.path.join(self.save_jpg_dir, 'json', 'cuboids_count.json'), 'w'), indent=4)
    
    def single_pro(self):
        self.task_func(self.train_list, 0)
        json.dump(self.count_dict['semseg_count'], open(os.path.join(self.save_jpg_dir, 'json', 'semseg_count.json'), 'w'), indent=4)
        json.dump(self.count_dict['cuboids_count'], open(os.path.join(self.save_jpg_dir, 'json', 'cuboids_count.json'), 'w'), indent=4)

    def task_func(self, sub_task, process_num):
        process_count_dict = {
            'semseg_count': {},
            'cuboids_count': {}
        }
        semseg_mark_dict = {}
        for sub in tqdm(sub_task):
            full_dir = os.path.join(self.data_dir, sub)
            semseg_dir = os.path.join(full_dir, 'annotations/semseg')
            for idx in range(80):
                semseg_data = pickle.load(gzip.open(os.path.join(semseg_dir, str(idx).zfill(2) + '.pkl.gz'))).values
                cuboids_data = pickle.load(gzip.open(os.path.join(full_dir, 'annotations/cuboids', str(idx).zfill(2) + '.pkl.gz'))).values
                process_count_dict['semseg_count'], semseg_mark_dict = self.count_sem_cat_num(
                    semseg_data, process_count_dict['semseg_count'], semseg_mark_dict, sub, idx
                )
                process_count_dict['cuboids_count'] = self.count_cuboids_cat_num(cuboids_data, process_count_dict['cuboids_count'])

        self.random_sample_vis(semseg_mark_dict)
        for key in process_count_dict['semseg_count']:
            if key not in self.count_dict['semseg_count']:
                self.count_dict['semseg_count'].update({key:0})
            self.count_dict['semseg_count'][key] += process_count_dict['semseg_count'][key]
        for key in process_count_dict['cuboids_count']:
            if key not in self.count_dict['cuboids_count']:
                self.count_dict['cuboids_count'].update({key:0})
            self.count_dict['cuboids_count'][key] += process_count_dict['cuboids_count'][key]
        
    
    def random_sample_vis(self, semseg_mark_dict):
        for label_name in tqdm(semseg_mark_dict):
            label_list = semseg_mark_dict[label_name]
            label_random_l = random.sample(label_list, 5)
            for label_random in label_random_l:
                sub = label_random.split('_')[0]
                idx = label_random.split('_')[1]
                lidar_data = pickle.load(gzip.open(os.path.join(self.data_dir, sub, 'lidar', idx + '.pkl.gz'))).values
                semseg_data = pickle.load(gzip.open(os.path.join(self.data_dir, sub, 'annotations/semseg', idx + '.pkl.gz'))).values
                label_num = self.value_get_key(self.labels, label_name)
                assert label_num is not None
                label_flag = np.ones(semseg_data.shape[0], dtype=bool)
                label_flag = np.logical_and(label_flag, semseg_data.flatten()==label_num)
                lidar_label_data = lidar_data[label_flag]
                self.vis(sub, idx, lidar_label_data, label_name)
    
    def vis(self, sub, idx, lidar_label_data, label_name):
        camera_list = ['back_camera', 'front_camera', 'front_left_camera', 'front_right_camera', 'left_camera', 'right_camera']
        for camera in camera_list:
            camera_data = Image.open(os.path.join(self.data_dir, sub, 'camera', camera, idx + '.jpg'))
            camera_pose = json.load(open(os.path.join(self.data_dir, sub, 'camera', camera, 'poses.json')))[int(idx)]
            camera_intrinsics = json.load(open(os.path.join(self.data_dir, sub, 'camera', camera, 'intrinsics.json')))
            camera_intrinsics = Intrinsics(fx=camera_intrinsics['fx'],
                                           fy=camera_intrinsics['fy'],
                                           cx=camera_intrinsics['cx'],
                                           cy=camera_intrinsics['cy'])
            projected_points2d, _, inner_indices = geometry.projection(lidar_points=lidar_label_data[:, :3], 
                                                                        camera_data=camera_data,
                                                                        camera_pose=camera_pose,
                                                                        camera_intrinsics=camera_intrinsics,
                                                                        filter_outliers=True)
            if projected_points2d.shape[0] == 0:continue
            img_c = np.array(camera_data)
            img = cv2.cvtColor(img_c,cv2.COLOR_RGB2BGR)
            img_ori = cv2.cvtColor(img_c,cv2.COLOR_RGB2BGR)
            for pts in projected_points2d:
                cv2.circle(img, (int(pts[0]), int(pts[1])), 1, (0,255,0), 4)
            if ' ' in label_name:
                label_name.replace(' ', '_')
            save_jpg_dir = os.path.join(self.save_jpg_dir, 'jpg', label_name, sub + '_' + idx, 'project_label')
            ori_save_jpg_dir = os.path.join(self.save_jpg_dir, 'jpg', label_name, sub + '_' + idx, 'ori')
            os.makedirs(ori_save_jpg_dir, exist_ok=True)
            os.makedirs(save_jpg_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_jpg_dir, camera + '.jpg'), img)
            cv2.imwrite(os.path.join(ori_save_jpg_dir, camera + '.jpg'), img_ori)

    
    def value_get_key(self, dict, value):
        for key in dict:
            if dict[key] == value:
                return key
        return None

    def count_sem_cat_num(self, semseg_data, semseg_count, semseg_mark_dict, sub, idx):
        semseg_data_unq = np.unique(semseg_data)
        for label in semseg_data_unq:
            if label == 0:
                continue
            label_num = np.where(semseg_data == label)[0].shape[0]
            label_name = self.labels[label]
            if label_name not in semseg_count:
                semseg_count.update({label_name: 0})
            if label_name not in semseg_mark_dict:
                semseg_mark_dict.update({label_name:[]})
            semseg_mark_dict[label_name].append(sub + '_' + str(idx).zfill(2))
            semseg_count[label_name] += label_num
        return semseg_count, semseg_mark_dict

    def count_cuboids_cat_num(self, cuboids_data, cuboids_count):
        label_data = cuboids_data[:, 1]
        label_unq = np.unique(label_data)
        for label in label_unq:
            label_num = np.where(cuboids_data == label)[0].shape[0]
            if label not in cuboids_count:
                cuboids_count.update({label:0})
            cuboids_count[label] += label_num
        return cuboids_count
    
    def count_all_cuboids(self):
        cuboids_count = {}
        paths = os.listdir(self.data_dir)
        for path in tqdm(paths):
            full_dir = os.path.join(self.data_dir, path, 'annotations', 'cuboids')
            for i in range(80):
                cuboids_data = pickle.load(gzip.open(os.path.join(full_dir, str(i).zfill(2) + '.pkl.gz'))).values
                label_data = cuboids_data[:, 1]
                label_unq = np.unique(label_data)
                for label in label_unq:
                    label_num = np.where(cuboids_data == label)[0].shape[0]
                    if label not in cuboids_count:
                        cuboids_count.update({label:0})
                    cuboids_count[label] += label_num
        json.dump(cuboids_count, open('{}/json/cuboids_count_all.json'.format(self.save_jpg_dir), 'w'), indent=4)

class NusceneseCount:
    def __init__(self, config_file, data_dir, save_jpg_dir, data_set='train'):
        self.yaml_conf = yaml.load(open(config_file), Loader=yaml.FullLoader)
        self.data_dir = data_dir
        self.save_jpg_dir = save_jpg_dir
        scenes = splits.train if data_set == 'train' else splits.val
        version = 'v1.0-trainval'
        self.dataset = data_set
        self.img_view = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        self.count_dict, self.label_dict = {}, {}

        self.nusc = NuScenes(version=version, dataroot=data_dir, verbose=True)
        self.get_available_scenes()
        available_scene_names = [s['name'] for s in self.available_scenes]
        scenes = list(filter(lambda x: x in available_scene_names, scenes))
        scenes = set([self.available_scenes[available_scene_names.index(s)]['token'] for s in scenes])
        self.get_path_infos_cam_lidar(scenes)
        print('Total %d scenes' % (len(self.token_list)))

    def task_func(self):
        self.count()
        if self.dataset == 'train':
            for label in tqdm(self.label_dict):
                random_label_token = random.sample(self.label_dict[label], min(len(self.label_dict[label]), 5))
                self.vis(random_label_token, label)

    def vis(self, token_list, label):
        for i in range(len(token_list)):
            lidar_data, annotated_data, lidar_sample_token = self.loadDataByIndex(token_list[i])
            label_flag = np.logical_and(np.ones(annotated_data.shape[0], dtype=bool), annotated_data[:, 0] == label)
            lidar_label_data = lidar_data[label_flag]
            for image_id in range(6):
                image, cam_sample_token = self.loadImage(token_list[i], image_id)
                cam_path, _, cam_intrinsic = self.nusc.get_sample_data(cam_sample_token)
                calib_infos = self.get_calib_infos(lidar_sample_token, cam_sample_token, cam_intrinsic)
                keep_idx, _, points_img = self.map_pointcloud_to_image(lidar_label_data[:, :3], (image.size[1], image.size[0]), calib_infos)
                points_img = np.ascontiguousarray(np.fliplr(points_img[keep_idx]))
                if points_img.shape[0] == 0: continue
                ori_img = cv2.imread(cam_path)
                save_dir = os.path.join(self.save_jpg_dir, 'jpg', self.yaml_conf['labels'][label], str(i))
                os.makedirs(os.path.join(save_dir, 'ori'), exist_ok=True)
                os.makedirs(os.path.join(save_dir, 'project_label'), exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, 'ori', self.img_view[image_id] + '.jpg'), ori_img)
                for pts in points_img:
                    cv2.circle(ori_img, (int(pts[1]), int(pts[0])), 1, (255,0,0), 4)
                cv2.imwrite(os.path.join(save_dir, 'project_label', self.img_view[image_id] + '.jpg'), ori_img)

    def vis_test(self, token):
        lidar_data, annotated_data, lidar_sample_token = self.loadDataByIndex(token)
        for image_id in range(6):
            image, cam_sample_token = self.loadImage(token, image_id)
            cam_path, _, cam_intrinsic = self.nusc.get_sample_data(cam_sample_token)
            calib_infos = self.get_calib_infos(lidar_sample_token, cam_sample_token, cam_intrinsic)
            keep_idx, depths, points_img = self.map_pointcloud_to_image(lidar_data[:, :3], (image.size[1], image.size[0]), calib_infos)
            points_img = np.ascontiguousarray(np.fliplr(points_img[keep_idx]))
            ori_img = cv2.imread(cam_path)
            for pts in points_img:
                cv2.circle(ori_img, (int(pts[1]), int(pts[0])), 1, (255,0,0), 4)
            cv2.imwrite('/home/qi.chao/c7/mycode/2dpass/work_dirs/nuscenes/bev_depth/jpg/test_1.jpg', ori_img)

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

        return mask, depths, points.T[:, :2]

    def get_calib_infos(self, lidar_sample_token, cam_sample_token, cam_intrinsic):
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
        return calib_infos

    def count(self):
        for token_dict in tqdm(self.token_list):
            _, annotated_data, _ = self.loadDataByIndex(token_dict)
            unique_anno_datas = np.unique(annotated_data)
            for label in unique_anno_datas:
                label_num = np.where(annotated_data == label)[0].shape[0]
                label_name = self.yaml_conf['labels'][label]
                if label_name not in self.count_dict:
                    self.count_dict.update({label_name: 0})
                self.count_dict[label_name] += label_num
                
                if label not in self.label_dict:
                    self.label_dict.update({label: []})
                if label_num >= 10:
                    self.label_dict[label].append(token_dict)
        os.makedirs(os.path.join(self.save_jpg_dir, 'json'), exist_ok=True)
        json.dump(self.count_dict, open(os.path.join(self.save_jpg_dir, 'json', '{}_semseg_count'.format(self.dataset)), 'w'), indent=4)
    
    def loadDataByIndex(self, token):
        lidar_sample_token = token['lidar_token']
        lidar_path = os.path.join(self.data_dir, self.nusc.get('sample_data', lidar_sample_token)['filename'])
        raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))

        lidarseg_path = os.path.join(self.data_dir, self.nusc.get('lidarseg', lidar_sample_token)['filename'])
        annotated_data = np.fromfile(lidarseg_path, dtype=np.uint8).reshape((-1, 1))  # label

        pointcloud = raw_data[:, :4]
        return pointcloud, annotated_data, lidar_sample_token

    def loadImage(self, token, image_id):
        cam_sample_token = token['cam_token'][image_id]
        cam = self.nusc.get('sample_data', cam_sample_token)
        image = Image.open(os.path.join(self.nusc.dataroot, cam['filename']))
        return image, cam_sample_token

    def get_available_scenes(self):
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
                self.token_list.append(
                    {'lidar_token': lidar_token,
                        'cam_token': cam_token}
                )
    

def pandaset_count():
    save_jpg_dir = 'work_dirs/pandaset/count'
    config_file = 'config/label_mapping/pandaset.yaml'
    data_dir = '/share/qi.chao/open_sor_data/pandaset/pandaset_0'
    count = PandasetCount(config_file, data_dir, save_jpg_dir)
    count.single_pro()

def nuscenese_count():
    save_jpg_dir = 'work_dirs/nuscenes/count'
    config_file = 'config/label_mapping/nuscenes.yaml'
    data_dir = '/share-global/xinli.xu/dataset/nuscenes'
    count = NusceneseCount(config_file, data_dir, save_jpg_dir)
    # count.task_func()
    count.vis_test(count.token_list[0])


nuscenese_count()
