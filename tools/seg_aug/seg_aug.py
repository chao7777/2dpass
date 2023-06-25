import cv2
import numpy as np
from seg_paste import SegPaste
import seg_utils
from tools.pandaset import geometry
from tools.pandaset.sensors import Intrinsics
import json
import os
import open3d
import yaml
import pickle
import gzip
from PIL import Image
import random
import pandas
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon,Point
import alphashape

class RealTimeAug:
    def __init__(self, config_file):
        self.yaml_conf = yaml.load(open(config_file), Loader=yaml.FullLoader)
        self.seg_paste = SegPaste(self.yaml_conf)
        self.filter_labels = [i for i in range(1, 13)]
        self.lidar_pcd = open3d.geometry.PointCloud()
        
    def task_func(self, tar_paste_info):
        for label in self.yaml_conf['seg_paste']:
            paste_num = self.get_paste_num(tar_paste_info, label)
            for i in range(paste_num):
                seg_seed_jpg_info = self.get_seg_seed_info(label)
                paste_loc = self.get_paste_loc(tar_paste_info, label, seg_seed_jpg_info)
                if paste_loc is None: continue
            return

    def get_paste_loc(self, tar_paste_info, label, seg_seed_jpg_info):
        filter_labels = self.filter_labels
        if label != 1:  filter_labels.remove(5)
        _, _, inner_indices = geometry.projection(lidar_points=tar_paste_info['lidar_data'][:, :3], 
                                                  camera_data=tar_paste_info['camera_data'],
                                                  camera_pose=tar_paste_info['calib_info']['camera_pose'],
                                                  camera_intrinsics=tar_paste_info['calib_info']['camera_intrinsics'],
                                                  filter_outliers=True)
        camera_seg_data = tar_paste_info['semseg'][inner_indices]
        camera_lidar_data = tar_paste_info['lidar_data'][inner_indices]
        filter_ground_lidar_data, _ = seg_utils.filter_lidar_data(camera_lidar_data, camera_seg_data, filter_labels)
        paste_label_lidar, _ = seg_utils.filter_lidar_data(camera_lidar_data, camera_seg_data, self.yaml_conf['paste2label'], filter=False)
        paste_label_lidar = self.filter_xyz_dis_lidar(paste_label_lidar, -50, 50, -50, 50)
        paste_label_lidar2ego2sph = seg_utils.lidar2ego2spherical(paste_label_lidar[:, :3], tar_paste_info['calib_info']['lidar_pose'])
        translate_center_e = self.get_translate_cen(paste_label_lidar2ego2sph, seg_seed_jpg_info, filter_ground_lidar_data, tar_paste_info, label)
        return translate_center_e


    def get_translate_cen(self, paste_label_lidar2ego2sph, seg_seed_jpg_info, filter_ground_lidar_data, tar_paste_info, label):
        filter_ground_lidar2ego_data = geometry.lidar_points_to_ego(filter_ground_lidar_data, tar_paste_info['calib_info']['lidar_pose'])
        
        random_center = random.sample(paste_label_lidar2ego2sph.tolist(), 3)
        translate_center_e = None
        for translate_center in random_center:
            seg_seed_center = geometry.world2spherical(seg_seed_jpg_info['center'])
            translate_dis = np.array(translate_center) - seg_seed_center
            seg_seed_spherical_data = geometry.world2spherical(seg_seed_jpg_info['seg_seed_lidar_data'])
            self.lidar_pcd.points = open3d.utility.Vector3dVector(seg_seed_spherical_data)
            self.lidar_pcd.translate(translate_dis, relative=True)

            seg_seed_tran_ego_data = geometry.spherical2world(np.array(self.lidar_pcd.points))
            if label == 1:
                filter_ground_less_2 = self.filter_xyz_dis_lidar(filter_ground_lidar2ego_data, z_max=2)
                tran_dis_less_2 = self.judge_lidar_overlap_dis(seg_seed_tran_ego_data, filter_ground_less_2, 'less_2')
                if tran_dis_less_2 is None: continue
                filter_ground_more_2 = self.filter_xyz_dis_lidar(filter_ground_lidar2ego_data, z_min=2)
                tran_dis_more_2 = self.judge_lidar_overlap_dis(seg_seed_tran_ego_data, filter_ground_more_2, 'more_2')
                if tran_dis_more_2 is None: continue
                tran_dis = tran_dis_more_2 if np.sum(np.square(tran_dis_more_2)) >= np.sum(np.square(tran_dis_less_2)) else tran_dis_less_2
            else:
                tran_dis =  self.judge_lidar_overlap_dis(seg_seed_tran_ego_data, filter_ground_lidar2ego_data)
                if tran_dis is None: continue
            translate_center_world = geometry.spherical2world(translate_center)
            translate_center_world[:, :2] += tran_dis
            translate_center = geometry.world2spherical(translate_center_world)
            translate_center_e = translate_center
            break
        return translate_center_e

            
    def judge_lidar_overlap_dis(self, seg_seed_tran_ego_data, camera_lidar2ego_data, dis):
        w, h, x, y = self.cal_wh(seg_seed_tran_ego_data, dis)
        x1, x2, y1, y2 = x - w / 2, x + w / 2, y - h / 2, y + h / 2
        overlap_pts = self.filter_xyz_dis_lidar(camera_lidar2ego_data, x1, x2, y1, y2)
        if overlap_pts.shape[0] == 0:
            return np.array([0, 0])
        repair_xy = self.repair_lidar_overlap_dis(overlap_pts, camera_lidar2ego_data, x1, x2, y1, y2)
        return repair_xy
    
    def repair_lidar_overlap_dis(self, overlap_pts, lidar_data, x1, x2, y1, y2):
        repair_xy = []
        left_x_dis = np.max(overlap_pts[:, 0]) - x1
        righe_x_dis = x2 - np.min(overlap_pts[:, 0])
        low_y_dis = np.max(overlap_pts[:, 1]) - y1
        high_y_dis = y2 - np.min(overlap_pts[:, 1])
        overlap_pts = self.filter_xyz_dis_lidar(lidar_data, x1 - righe_x_dis, x2 - righe_x_dis, y1 - high_y_dis, y2 - high_y_dis)
        if overlap_pts.shape[0] == 0 and x1 - righe_x_dis > np.min(lidar_data[:, 0]) and y1 - high_y_dis > np.min(lidar_data[:, 1]):
            repair_xy.append([-righe_x_dis, -high_y_dis])
        overlap_pts = self.filter_xyz_dis_lidar(lidar_data, x1 + left_x_dis, x2 + left_x_dis, y1 - high_y_dis, y2 - high_y_dis)
        if overlap_pts.shape[0] == 0 and x2 + left_x_dis < np.max(lidar_data[:, 0]) and y1 - high_y_dis > np.min(lidar_data[:, 1]):
            repair_xy.append([left_x_dis, -high_y_dis])
        overlap_pts = self.filter_xyz_dis_lidar(lidar_data, x1 + left_x_dis, x2 + left_x_dis, y1 + low_y_dis, y2 + low_y_dis)
        if overlap_pts.shape[0] == 0 and x2 + left_x_dis < np.max(lidar_data[:, 0]) and y2 + low_y_dis < np.max(lidar_data[:, 1]):
            repair_xy.append([left_x_dis, low_y_dis])
        overlap_pts = self.filter_xyz_dis_lidar(lidar_data, x1 - righe_x_dis, x2 - righe_x_dis, y1 + low_y_dis, y2 + low_y_dis)
        if overlap_pts.shape[0] == 0 and x1 - righe_x_dis > np.min(lidar_data[:, 0]) and y2 + low_y_dis < np.max(lidar_data[:, 1]):
            repair_xy.append([-righe_x_dis, low_y_dis])
        if len(repair_xy) == 0:
            return None
        repair_xy = np.array(repair_xy)
        repair_xy_square = np.sum(np.square(repair_xy), axis=1)
        max_index = np.where(repair_xy_square == np.max(repair_xy_square))[0][0]
        return repair_xy[max_index]

    def filter_xyz_dis_lidar(self, lidar_data, x_min=-100, x_max=100, y_min=-100, y_max=100, z_min=-50, z_max=50, filter=True):
        filter_flag = np.ones(lidar_data.shape[0], dtype=bool)
        condition_x = (lidar_data[:, 0] > x_min) & (lidar_data[:, 0] < x_max) 
        condition_y = (lidar_data[:, 1] > y_min) & (lidar_data[:, 1] < y_max)
        condition_z = (lidar_data[:, 2] > z_min) & (lidar_data[:, 2] < z_max)
        filter_flag = np.logical_and(filter_flag, condition_x & condition_y & condition_z)
        if not filter:
            filter_flag = ~filter_flag
        return lidar_data[filter_flag]
        
    
    def get_paste_num(self, tar_paste_info, label):
        _, _, inner_indices = geometry.projection(lidar_points=tar_paste_info['lidar_data'][:, :3], 
                                                  camera_data=tar_paste_info['camera_data'],
                                                  camera_pose=tar_paste_info['calib_info']['camera_pose'],
                                                  camera_intrinsics=tar_paste_info['calib_info']['camera_intrinsics'],
                                                  filter_outliers=True)
        camera_seg_data = tar_paste_info['label_seg'][inner_indices]
        label_seg_data_len = np.where(camera_seg_data == label)[0].shape[0]
        label_ratio = label_seg_data_len / camera_seg_data.shape[0]
        if label == 1:
            if label_ratio >= 0.3: return 0
            elif 0.1 < label_ratio < 0.3: return self.num_rand(0, 1, 0.4)
            elif 0 < label_ratio <= 0.1 : return self.num_rand(1, 2, 0.4)
            elif label_ratio == 0: return self.num_rand(2, 3, 0.6)
        
        elif label == 13:
            return 0 if label_ratio != 0 else self.num_rand(0, 1, 0.4)
        
        elif label == 8 or label == 9:
            return 1 if label_ratio == 0 else self.num_rand(0, 1, 0.3)
        
        elif label == 20 or label == 12:
            return self.num_rand(1, 2, 0.7) if label_ratio == 0 else self.num_rand(1, 2, 0.3)
        
        elif label == 18:
            return 0 if label_ratio != 0 else self.num_rand(0, 1, 0.3)

        elif label == 10 or label == 11 or label == 16:
            return self.num_rand(1, 2, 0.3)
        
        elif label == 14:
            return random.randint(1, 4)
        
        else:
            return 0

    def get_seg_seed_info(self, label):
        label_name = self.yaml_conf['labels_seg'][label]
        assert label_name is not None
        seg_seed_jpg_path = random.sample(self.seg_paste.all_seg_seed_jpg[label_name], 1)
        seg_seed_jpg_info = self.seg_paste.get_seed_jpg_info(seg_seed_jpg_path[0])
        lidar2ego_data = geometry.lidar_points_to_ego(seg_seed_jpg_info['seg_seed_lidar_data'], seg_seed_jpg_info['lidar_pose'])
        x = (np.min(lidar2ego_data[:, 0]) + np.max(lidar2ego_data[:, 0])) / 2
        y = (np.min(lidar2ego_data[:, 1]) + np.max(lidar2ego_data[:, 1])) / 2
        z = np.min(lidar2ego_data[:, 2])
        seg_seed_jpg_info.update({
            'jpg_path': seg_seed_jpg_path[0],
            'seg_seed_lidar_data': lidar2ego_data,
            'center': np.array((x, y, z))
        })

        return seg_seed_jpg_info
    
    def cal_wh(self, lidar2ego_data, dis):
        z_flag = np.ones(lidar2ego_data.shape[0], dtype=bool)
        if dis == 'less_2':
            z_dis = np.logical_and(z_flag, lidar2ego_data[:, 2] <= 2)
        elif dis == 'more_2':
            z_dis = np.logical_and(z_flag, lidar2ego_data[:, 2] > 2)
        else:
            z_dis = np.copy(z_flag)
        z_dis = lidar2ego_data[z_dis]
        w = np.max(z_dis[:, 0]) - np.min(z_dis[:, 0])
        h = np.max(z_dis[:, 1]) - np.min(z_dis[:, 0])
        x = (np.max(z_dis[:, 0]) + np.min(z_dis[:, 0])) / 2
        y = (np.max(z_dis[:, 1]) + np.min(z_dis[:, 0])) / 2
        return w, h, x, y

    def num_rand(self, min, max, max_pro):
        random_num = random.randint(1, 10)
        return max if random_num <= max_pro * 10 else min


class LocalAug:
    def __init__(self, config_file):
        self.yaml_conf = yaml.load(open(config_file), Loader=yaml.FullLoader)
        self.seg_paste = SegPaste(self.yaml_conf)
        self.filter_labels = [i for i in range(1, 13)]
        self.local_remove_labels = [13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 30, 31, 32, 33, 37, 40]
        self.lidar_pcd = open3d.geometry.PointCloud()
        self.pandaset_dir = self.yaml_conf['pandaset_dir']
        self.save_dir = '/share/qi.chao/open_sor_data/pandaset/pandaset_seg'
        self.camera_list = ['back_camera', 'front_camera', 'front_left_camera', 'front_right_camera', 'left_camera', 'right_camera']
        
    def seg_paste_func(self):
        sub_dirs = os.path.join(self.save_dir)
        for sub in sub_dirs:
            for idx in range(80):
                lidar_info = self.get_lidar_info(self.save_dir, sub, str(idx).zfill(2))
                for camera_name in self.camera_list:
                    camera_info = self.get_camera_info(self.save_dir, sub, str(idx).zfill(2), camera_name)

    def gen_blank_scene(self):
        sub_dirs = random.sample(self.yaml_conf['train_list'], 10)
        for sub in tqdm(sub_dirs):
            for idx in tqdm(range(80)):
                save_dict = self.filter_lidar(sub, str(idx).zfill(2))
                self.save_info(save_dict)
    
    def filter_lidar(self, sub, idx):
        lidar_info = self.get_lidar_info(self.pandaset_dir, sub, idx)
        reserve_lidar, reserve_semseg = seg_utils.filter_lidar_data(lidar_info['lidar_data'], lidar_info['semseg_data'], self.local_remove_labels, filter=True)
        save_dict = {
            'lidar_data': reserve_lidar,
            'semseg_data': reserve_semseg,
            'lidar_pose': lidar_info['lidar_pose'],
            'lidar_timestamps': lidar_info['timestamps'],
            'sub': sub,
            'idx': idx
        }
        for camera_name in self.camera_list:
            camera_info = self.get_camera_info(self.pandaset_dir, sub, idx, camera_name)
            _, _, inner_indices = geometry.projection(lidar_points=lidar_info['lidar_data'][:, :3], 
                                                      camera_data=camera_info['camera_data'],
                                                      camera_pose=camera_info['camera_pose'][int(idx)],
                                                      camera_intrinsics=camera_info['camera_intrinsics_i'],
                                                      filter_outliers=True)
            camera_lidar_data = lidar_info['lidar_data'][inner_indices]
            camera_semseg_data = lidar_info['semseg_data'][inner_indices]
            img = self.remove_camera_pts(camera_info, camera_lidar_data, camera_semseg_data, idx)
            camera_info.update({
                'img': img
            })
            save_dict.update({
                camera_name: camera_info
            })
        return save_dict
    
    def save_info(self, save_dict):
        lidar_dir = os.path.join(self.save_dir, save_dict['sub'], 'lidar')
        semseg_dir = os.path.join(self.save_dir, save_dict['sub'], 'annotations', 'semseg')
        camera_dirs = [os.path.join(self.save_dir, save_dict['sub'], 'camera', camera_name) for camera_name in self.camera_list]
        self.mkdir([lidar_dir, semseg_dir])
        self.mkdir(camera_dirs)
        pickle.dump(save_dict['lidar_data'], gzip.open(os.path.join(lidar_dir, save_dict['idx'] + '.pkl.gz'), 'wb'))
        pickle.dump(save_dict['semseg_data'], gzip.open(os.path.join(semseg_dir, save_dict['idx'] + '.pkl.gz'), 'wb'))
        json.dump(save_dict['lidar_pose'], open(os.path.join(lidar_dir, 'poses.json'), 'w'))
        json.dump(save_dict['lidar_timestamps'], open(os.path.join(lidar_dir, 'timestamps.json'), 'w'))
        for camera_name in self.camera_list:
            camera_dir = os.path.join(self.save_dir, save_dict['sub'], 'camera', camera_name)
            cv2.imwrite(os.path.join(camera_dir, save_dict['idx'] + '.jpg'), save_dict[camera_name]['img'])
            json.dump(save_dict[camera_name]['camera_pose'], open(os.path.join(camera_dir, 'poses.json'), 'w'))
            json.dump(save_dict[camera_name]['camera_intrinsics'], open(os.path.join(camera_dir, 'intrinsics.json'), 'w'))
            json.dump(save_dict[camera_name]['timestamps'], open(os.path.join(camera_dir, 'timestamps.json'), 'w'))

    def mkdir(self, dir_paths):
        for dir_path in dir_paths:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
    
    def remove_camera_pts(self, camera_info, lidar_data, semseg_data, idx):
        img = np.array(camera_info['camera_data'])[..., ::-1]
        for label in self.local_remove_labels:
            label_lidar_data, _ = seg_utils.filter_lidar_data(lidar_data, semseg_data, [label], filter=False)
            if label_lidar_data.shape[0] == 0: continue
            clustering = DBSCAN(eps=1.5, min_samples=4).fit(label_lidar_data[:, :3])
            cluster_label = clustering.labels_
            for i in range(max(cluster_label)+1):
                cluster_labels_flag = np.ones(label_lidar_data.shape[0], dtype=bool)
                cluster_labels_flag = np.logical_and(cluster_labels_flag, np.array(cluster_label)==i)
                cluster_label_data = label_lidar_data[cluster_labels_flag]
                projected_points2d, _, inner_indices = geometry.projection(lidar_points=cluster_label_data[:, :3], 
                                                                            camera_data=camera_info['camera_data'],
                                                                            camera_pose=camera_info['camera_pose'][int(idx)],
                                                                            camera_intrinsics=camera_info['camera_intrinsics_i'],
                                                                            filter_outliers=True)
                alpha_shape=alphashape.alphashape(projected_points2d.tolist(), 0)
                min_x, max_x = np.min(projected_points2d[:, 0]), np.max(projected_points2d[:, 0])
                min_y, max_y = np.min(projected_points2d[:, 1]), np.max(projected_points2d[:, 1])
                alpha_shape = list(alpha_shape.exterior.coords)
                polygon = Polygon(alpha_shape)
                for x_idx in range(int(min_x - 1), int(max_x + 1)):
                    for y_idx in range(int(min_y - 1), int(max_y + 1)):
                        if polygon.contains(Point(x_idx, y_idx)):
                            img[y_idx, x_idx] = (0, 0, 0)
        return img    
    
    def get_lidar_info(self, fir_dir, sub, idx):
        lidar_data = pickle.load(gzip.open(os.path.join(fir_dir, sub, 'lidar', idx + '.pkl.gz'))).values
        semseg_data = pickle.load(gzip.open(os.path.join(fir_dir, sub, 'annotations/semseg', idx + '.pkl.gz'))).values
        lidar_pose = json.load(open(os.path.join(fir_dir, sub, 'lidar/poses.json')))
        timestamps = json.load(open(os.path.join(fir_dir, sub, 'lidar/timestamps.json')))
        lidar_info = {
            'lidar_data': lidar_data,
            'semseg_data': semseg_data,
            'lidar_pose': lidar_pose,
            'timestamps': timestamps
        }
        return lidar_info
    
    def get_camera_info(self, fir_dir, sub, idx, camera_name):
        camera_data = Image.open(os.path.join(fir_dir, sub, 'camera', camera_name, idx + '.jpg'))
        camera_pose = json.load(open(os.path.join(fir_dir, sub, 'camera/{}/poses.json'.format(camera_name))))
        camera_intrinsics = json.load(open(os.path.join(fir_dir, sub, 'camera/{}/intrinsics.json'.format(camera_name))))
        timestamps = json.load(open(os.path.join(fir_dir, sub, 'camera/{}/timestamps.json'.format(camera_name))))
        camera_intrinsics_i = Intrinsics(fx=camera_intrinsics['fx'],
                                        fy=camera_intrinsics['fy'],
                                        cx=camera_intrinsics['cx'],
                                        cy=camera_intrinsics['cy'])
        camera_info = {
            'camera_data': camera_data,
            'camera_pose': camera_pose,
            'timestamps': timestamps,
            'camera_intrinsics': camera_intrinsics,
            'camera_intrinsics_i': camera_intrinsics_i
        }
        return camera_info

def vis(tar_paste_info):
    projec_2d, _, inner_indices = geometry.projection(lidar_points=tar_paste_info['lidar_data'][:, :3], 
                                                  camera_data=tar_paste_info['camera_data'],
                                                  camera_pose=tar_paste_info['calib_info']['camera_pose'],
                                                  camera_intrinsics=tar_paste_info['calib_info']['camera_intrinsics'],
                                                  filter_outliers=True)
    camera_seg_data = tar_paste_info['semseg'][inner_indices]
    camera_lidar_data = tar_paste_info['lidar_data'][inner_indices]
    paste_label = np.ones(camera_seg_data.shape[0], dtype=bool)
    paste_label = np.logical_and(paste_label, (camera_seg_data[:, 0] == 5))
    road_projec_2d = projec_2d[paste_label].tolist()
    img = np.array(tar_paste_info['camera_data'])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    random_road = random.sample(road_projec_2d, 10)
    for pts in road_projec_2d:
        cv2.circle(img, (int(pts[0]), int(pts[1])), 3, (0,255,0), 4)
    cv2.imwrite('work_dirs/test.jpg', img)

def test():
    seg_seed_jpg_path = '/share/qi.chao/open_sor_data/pandaset/seed_jpg/grabcut/removebg/053/00/back_camera/Vegetation/Vegetation/label_1527_291_1829_636.jpg'
    config_file = 'config/label_mapping/pandaset.yaml'
    dataset_yaml = yaml.safe_load(open(config_file))
    pandaset_dir = '/share/qi.chao/open_sor_data/pandaset/pandaset_0'
    sub, idx, camera_name = '013', '25', 'back_camera'
    lidar_data = pickle.load(gzip.open(os.path.join(pandaset_dir, sub, 'lidar', idx + '.pkl.gz'))).values
    # lidar_data = pickle.load(gzip.open('work_dirs/pandaset/jpg/013_25_back_camera/repair_25.pkl.gz')).values
    semseg_data = pickle.load(gzip.open(os.path.join(pandaset_dir, sub, 'annotations/semseg', idx + '.pkl.gz'))).values
    cuboids_data = pickle.load(gzip.open(os.path.join(pandaset_dir, sub, 'annotations/cuboids', idx + '.pkl.gz'))).values
    camera_data = Image.open(os.path.join(pandaset_dir, sub, 'camera', camera_name, idx + '.jpg'))
    camera_pose = json.load(open(os.path.join(pandaset_dir, sub, 'camera/{}/poses.json'.format(camera_name))))[int(idx)]
    lidar_pose = json.load(open(os.path.join(pandaset_dir, sub, 'lidar/poses.json')))[int(idx)]
    camera_intrinsics = json.load(open(os.path.join(pandaset_dir, sub, 'camera/{}/intrinsics.json'.format(camera_name))))
    camera_intrinsics = Intrinsics(fx=camera_intrinsics['fx'],
                                    fy=camera_intrinsics['fy'],
                                    cx=camera_intrinsics['cx'],
                                    cy=camera_intrinsics['cy'])
    calib_info = {
        'camera_pose': camera_pose,
        'lidar_pose': lidar_pose,
        'camera_intrinsics': camera_intrinsics,
        'sub_dir': sub,
        'idx': idx,
        'camera_name': camera_name,
    }
    data_dict = {
        'lidar_data': lidar_data,
        'cuboids_data': cuboids_data,
        'calib_info': calib_info,
        'semseg': semseg_data,
        'label_seg': np.vectorize(dataset_yaml['learning_map'].__getitem__)(semseg_data),
        'camera_data': camera_data
    }
    # seg_paste_aug = SegPaste(config_file)
    # seg_paste_aug.seg_paste(seg_seed_jpg_path, data_dict)
    # real_time_aug = RealTimeAug(config_file)
    # real_time_aug.task_func(data_dict)
    # local_aug = LocalAug(config_file)
    # local_aug.gen_blank_scene()
    vis(data_dict)

if __name__ == '__main__':
    test()