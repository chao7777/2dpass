import pickle
import gzip
import numpy as np
import cv2
from tqdm import tqdm
import yaml
import json
from get_seed import MultiProcess
import os
from PIL import Image
from tools.pandaset import geometry
from tools.pandaset.sensors import Intrinsics
import random


class Count:
    def __init__(self, config_file, dir, save_jpg_dir):
        yaml_conf = yaml.load(open(config_file), Loader=yaml.FullLoader)
        self.train_list, self.val_list = yaml_conf['train_list'], yaml_conf['val_list']
        self.labels = yaml_conf['labels']
        self.count_dict = {
            'semseg_count': {},
            'cuboids_count': {}
        }
        self.dir = dir
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
            full_dir = os.path.join(self.dir, sub)
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
                lidar_data = pickle.load(gzip.open(os.path.join(self.dir, sub, 'lidar', idx + '.pkl.gz'))).values
                semseg_data = pickle.load(gzip.open(os.path.join(self.dir, sub, 'annotations/semseg', idx + '.pkl.gz'))).values
                label_num = self.value_get_key(self.labels, label_name)
                assert label_num is not None
                label_flag = np.ones(semseg_data.shape[0], dtype=bool)
                label_flag = np.logical_and(label_flag, semseg_data.flatten()==label_num)
                lidar_label_data = lidar_data[label_flag]
                self.vis(sub, idx, lidar_label_data, label_name)
    
    def vis(self, sub, idx, lidar_label_data, label_name):
        camera_list = ['back_camera', 'front_camera', 'front_left_camera', 'front_right_camera', 'left_camera', 'right_camera']
        for camera in camera_list:
            camera_data = Image.open(os.path.join(self.dir, sub, 'camera', camera, idx + '.jpg'))
            camera_pose = json.load(open(os.path.join(self.dir, sub, 'camera', camera, 'poses.json')))[int(idx)]
            camera_intrinsics = json.load(open(os.path.join(self.dir, sub, 'camera', camera, 'intrinsics.json')))
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
            if not os.path.exists(ori_save_jpg_dir):
                os.makedirs(ori_save_jpg_dir)
            if not os.path.exists(save_jpg_dir):
                os.makedirs(save_jpg_dir)
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


def check():
    dir = '/share/qi.chao/open_sor_data/pandaset/pandaset_0'
    sub_dirs = os.listdir(dir)
    for sub_dir in tqdm(sub_dirs):
        full_dir = os.path.join(dir, sub_dir, 'annotations/cuboids')
        paths = os.listdir(full_dir)
        for i in range(80):
            if str(i).zfill(2) + '.pkl.gz' not in paths:
                print(full_dir)
                print(i)
        if '80.pkl.gz' in paths:
            print(full_dir)

def count_cuboids():
    cuboids_count = {}
    dir = '/share/qi.chao/open_sor_data/pandaset/pandaset_0'
    paths = os.listdir(dir)
    for path in tqdm(paths):
        full_dir = os.path.join(dir, path, 'annotations', 'cuboids')
        for i in range(80):
            cuboids_data = pickle.load(gzip.open(os.path.join(full_dir, str(i).zfill(2) + '.pkl.gz'))).values
            label_data = cuboids_data[:, 1]
            label_unq = np.unique(label_data)
            for label in label_unq:
                label_num = np.where(cuboids_data == label)[0].shape[0]
                if label not in cuboids_count:
                    cuboids_count.update({label:0})
                cuboids_count[label] += label_num
    json.dump(cuboids_count, open('work_dirs/pandaset/count/json/cuboids_count_all.json', 'w'), indent=4)


def main():
    save_jpg_dir = 'work_dirs/pandaset/count'
    config_file = 'config/label_mapping/pandaset.yaml'
    dir = '/share/qi.chao/open_sor_data/pandaset/pandaset_0'
    count = Count(config_file, dir, save_jpg_dir)
    count.single_pro()

main()
