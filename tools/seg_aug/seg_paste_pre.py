import numpy as np
from tqdm import tqdm
import cv2
import open3d
import os
from tools.pandaset import geometry
from tools.pandaset.sensors import Intrinsics
from PIL import Image
import yaml
import json
import pickle
import gzip
import random
import shutil
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon,Point
import alphashape
from get_seed import write_data, MultiProcess, read_data

class GetSeed:
    def __init__(self, dir, config_file, save_dir):
        yaml_conf = yaml.load(open(config_file), Loader=yaml.FullLoader)
        self.dir, self.save_dir = dir, save_dir
        self.train_list = yaml_conf['train_list']
        self.labels = yaml_conf['labels']
        self.labels_sim = yaml_conf['labels_sim']
        self.learning_map = yaml_conf['learning_map']
        self.seg_paste = yaml_conf['seg_paste']
        self.camera_list = ['back_camera', 'front_camera', 'front_left_camera', 'front_right_camera', 'left_camera', 'right_camera']
            

    def task_func(self, sub_task, process_num):
        for sub in tqdm(sub_task):
            sub_dir = os.path.join(self.dir, sub)
            for idx in range(80):
                # semseg_data = pickle.load(gzip.open(os.path.join(semseg_dir, str(idx).zfill(2) + '.pkl.gz'))).values
                pass
    
    def get_ori_label(self, label_num):
        ori_label_num_l = []
        for key in self.learning_map:
            if self.learning_map[key] == label_num:
                ori_label_num_l.append(key)
        return ori_label_num_l

    def get_cluster_point(self, label_data, cluster_label, camera_info_dict, label, ori_label):
        save_jpg_dir = os.path.join(self.save_dir, 'jpg', camera_info_dict['sub'], str(camera_info_dict['idx']).zfill(2), 
                                    camera_info_dict['camera_name'], self.labels_sim[label], self.labels[ori_label])
        save_lidar_dir = os.path.join(self.save_dir, 'lidar', camera_info_dict['sub'], str(camera_info_dict['idx']).zfill(2), 
                                    camera_info_dict['camera_name'], self.labels_sim[label], self.labels[ori_label])
        if not os.path.exists(save_jpg_dir):
            os.makedirs(save_jpg_dir)
        if not os.path.exists(save_lidar_dir):
            os.makedirs(save_lidar_dir)
        img = np.array(camera_info_dict['camera_data'])[...,::-1]
        for i in range(max(cluster_label)+1):
            cluster_labels_flag = np.ones(label_data.shape[0], dtype=bool)
            cluster_labels_flag = np.logical_and(cluster_labels_flag, np.array(cluster_label)==i)
            cluster_label_data = label_data[cluster_labels_flag]
            if np.min(cluster_label_data[:,2]) > 0.2:continue
            projected_points2d, _, _ = geometry.projection(lidar_points=cluster_label_data[:, :3], 
                                                            camera_data=camera_info_dict['camera_data'],
                                                            camera_pose=camera_info_dict['camera_pose'],
                                                            camera_intrinsics=camera_info_dict['camera_intrinsics'],
                                                            filter_outliers=True)
            projected_points2d = projected_points2d.astype(np.uint32)
            left, right = min(projected_points2d[:,0]) - 2, max(projected_points2d[:,0]) + 2
            low, high = min(projected_points2d[:,1]) - 30 if label == 1 else 2, max(projected_points2d[:,1]) + 2
            left, right = max(0, left), min(img.shape[1], right)
            low, high = max(0, low), min(img.shape[0], high)
            mid_x, mid_y = (left+right) // 2, (low+high) // 2
            cluster_label_img = img[low:high, left:right]
            cv2.imwrite(os.path.join(save_jpg_dir, 'label_{}_{}_{}_{}.jpg'.format(left, low, right, high)), cluster_label_img)
            write_data(cluster_label_data, save_lidar_dir, 'label_{}_{}_{}_{}'.format(left, low, right, high))

        
    def lidar_project_2d(self, sub, idx):
        lidar_data = pickle.load(gzip.open(self.dir, sub, 'lidar', str(idx).zfill(2) + '.pkl.gz')).values
        semseg_data = pickle.load(gzip.open(self.dir, sub, 'annotations', 'semseg', str(idx).zfill(2) + '.pkl.gz')).values
        assert lidar_data.shape[0] == semseg_data.shape[0]
        for camera_name in self.camera_list:
            camera_data = Image.open(os.path.join(self.dir, sub, 'camera', camera_name, idx + '.jpg'))
            camera_pose = json.load(open(os.path.join(self.dir, sub, 'camera', camera_name, 'poses.json')))[int(idx)]
            camera_intrinsics = json.load(open(os.path.join(self.dir, sub, 'camera', camera_name, 'intrinsics.json')))
            camera_intrinsics = Intrinsics(fx=camera_intrinsics['fx'],
                                           fy=camera_intrinsics['fy'],
                                           cx=camera_intrinsics['cx'],
                                           cy=camera_intrinsics['cy'])
            camera_info_dict = {
                'camera_data': camera_data,
                'camera_name': camera_name,
                'camera_intrinsics': camera_intrinsics,
                'camera_pose': camera_pose,
                'sub_dir': sub,
                'idx': idx
            }
            _, _, inner_indices = geometry.projection(lidar_points=lidar_data[:, :3], 
                                                        camera_data=camera_data,
                                                        camera_pose=camera_pose,
                                                        camera_intrinsics=camera_intrinsics,
                                                        filter_outliers=True)
            camera_semseg = semseg_data[inner_indices].flatten()
            xyz = lidar_data[inner_indices]
            for label in self.seg_paste:
                ori_label_l = self.get_ori_label(label)
                for ori_label in ori_label_l:
                    label_bool = np.ones(camera_semseg.shape[0], dtype=bool)
                    label_bool = np.logical_and(label_bool, camera_semseg==ori_label)
                    label_data = xyz[label_bool]
                    if label_data.shape[0] < 20:continue
                    clustering = DBSCAN(eps=1.5, min_samples=15).fit(label_data[:, :3])
                    # lidar_pcd.points = open3d.utility.Vector3dVector(label_data)
                    # lidar_pcd.translate((-5, 0, 0), relative=True)
                    cluster_label = clustering.labels_
                    self.get_cluster_point(label_data,
                                        cluster_label,
                                        camera_info_dict,
                                        label,
                                        ori_label)


def remove(target):
    if not os.path.exists(target) or ' ' in target:
        return
    if os.path.isdir(target):
        shutil.rmtree(target, ignore_errors=True)
    elif os.path.isfile(target):
        os.remove(target)

def remove_lidar_file(jpg_dir, lidar_dir):
    for root, dir, files in os.walk(lidar_dir):
        for file in files:
            lidar_data_path = os.path.join(root, file)
            jpg_path = os.path.join(jpg_dir, lidar_data_path.split(lidar_dir, 1) + '.jpg')
            if not os.path.exists(jpg_path):
                remove(lidar_data_path)

class Grabcut:
    def __init__(self, dir, save_dir, use_sam=False):
        self.dir, self.save_dir = dir, save_dir
        self.pandaset_dir = '/share/qi.chao/open_sor_data/pandaset/pandaset_0'
        self.lidar_dir = os.path.join(self.dir.split('/')[:-1], 'lidar')
        self.use_sam = use_sam
        if use_sam:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
            sam_checkpoint = "/share/qi.chao/open_sor_data/checkpoint/sam_vit_h_4b8939.pth"
            device = "cuda"
            model_type = "default"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            self.mask_generator = SamAutomaticMaskGenerator(sam)

    def get_all_jpg(self):
        self.all_jpg_paths = []
        for root, dir, files in self.dir:
            for file in files:
                if not file.endwith('.jpg'): continue
                self.all_jpg_paths.append(os.path.join(root, file))

    def grabcut(self, jpg_path):
        jpg_path_split = jpg_path.split('/')
        lidar_data_path = os.path.join(self.lidar_dir, '/'.join(jpg_path_split[-6:-1]), jpg_path_split[-1][:-4])
        lidar_data = read_data(lidar_data_path)
        sub, idx, camera_name, label, ori_label, jpg_name = jpg_path_split[-6:]
        save_grabcut_dir = os.path.join(self.save_dir, 'opencv', '/'.join(jpg_path_split[-6:-1]))
        if not os.path.exists(save_grabcut_dir): 
            os.makedirs(save_grabcut_dir)
        ori_jpg_path = os.path.join(self.pandaset_dir, sub, 'camera', camera_name, idx + '.jpg')
        x_min, y_min, x_max, y_max = jpg_name.split('_')[1:]
        ori_img = cv2.imread(ori_jpg_path)
        rect = (x_min,y_min, x_max - x_min, y_max - y_min)

        project_2d = self.get_project_2d(sub, camera_name, idx, lidar_data)
        img_g = self.opencv(label, ori_img, rect, project_2d)
        cv2.imwrite(os.path.join(save_grabcut_dir, jpg_name), img_g)
        
        if self.use_sam and label == 1:
            save_grabcut_dir = os.path.join(self.save_dir, 'sam', '/'.join(jpg_path_split[-6:-1]))
            if not os.path.exists(save_grabcut_dir): 
                os.makedirs(save_grabcut_dir)
            mask = self.sam_mask(ori_img, project_2d)
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            res = cv2.bitwise_and(ori_img, ori_img, mask=mask_gray)
            cv2.imwrite(os.path.join(save_grabcut_dir, jpg_name), res)

    def get_project_2d(self, sub, camera_name, idx, lidar_data):
        camera_data = Image.open(os.path.join(self.pandaset_dir, sub, 'camera', camera_name, idx + '.jpg'))
        camera_pose = json.load(open(os.path.join(self.pandaset_dir, sub, 'camera', camera_name, 'poses.json')))[int(idx)]
        camera_intrinsics = json.load(open(os.path.join(self.pandaset_dir, sub, 'camera', camera_name, 'intrinsics.json')))
        camera_intrinsics = Intrinsics(fx=camera_intrinsics['fx'],
                                        fy=camera_intrinsics['fy'],
                                        cx=camera_intrinsics['cx'],
                                        cy=camera_intrinsics['cy'])
        projected_points2d, _, inner_indices = geometry.projection(lidar_points=lidar_data[:, :3], 
                                                                    camera_data=camera_data,
                                                                    camera_pose=camera_pose,
                                                                    camera_intrinsics=camera_intrinsics,
                                                                    filter_outliers=True)
        return projected_points2d
            

    def opencv(self, label, ori_img, rect, projected_points2d):
        if label == 1:
            mask = np.zeros(ori_img.shape[:2], dtype=np.uint8)
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
            (mask, bgModel, fgModel) = cv2.grabCut(ori_img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask==0)|(mask==2),0,1).astype('uint8')
            img_g = ori_img * mask2[:,:,np.newaxis]
        else:
            alpha_shape=alphashape.alphashape(projected_points2d.tolits(),0)
            alpha_shape = list(alpha_shape.exterior.coords)
            polygon = Polygon(alpha_shape)
            mask = np.zeros(ori_img.shape[:2], np.uint8)
            for x_idx in range(mask.shape[1]):
                for y_idx in range(mask.shape[0]):
                    if polygon.contains(Point(x_idx, y_idx)):
                        mask[y_idx, x_idx] = 1
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
            (mask, bgModel, fgModel) = cv2.grabCut(ori_img, mask ,None, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)
            mask2 = np.where((mask==0)|(mask==2), 0, 1).astype('uint8')
            img_g = ori_img * mask2[:,:,np.newaxis]
        return img_g
    
    def sam_mask(self, ori_img, project_2d):
        masks = self.mask_generator.generate(ori_img)
        sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
        mask = np.zeros((ori_img.shape[0], ori_img.shape[1], 3))
        random_project_2d = random.sample(project_2d, min(len(project_2d), 10))

        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.random.random((1, 3))*255
            color_mask = color_mask.tolist()[0]
            if self.judge_mask(m, random_project_2d): continue
            m_true = np.where(m == True)
            m_true = np.dstack(m_true)
            for pts in m_true[0]:
                pts_l = pts.tolist()
                mask[pts_l[0], pts_l[1]] = [255,255,255]
        return mask

    def judge_mask(self, m, project_2d):
        num = 0
        for pts in project_2d:
            if m[pts[1], pts[0]] == False:
                num += 1
        return True if num > len(project_2d) // 2 else False
