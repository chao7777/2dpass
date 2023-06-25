import cv2
import open3d
from tools.pandaset import geometry
from tools.pandaset.sensors import Intrinsics
from tqdm import tqdm
import numpy as np
import yaml
import random
import json
from PIL import Image
from sklearn.cluster import DBSCAN
import os
from shapely.geometry import Polygon,Point
import alphashape
from seg_utils import read_data, spherical2ego2lidar, lidar2ego2spherical

def average_brightness(img):
    height, width = img.shape[0], img.shape[1]

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 提取出v通道信息
    v_day = cv2.split(hsv_img)[2]
    # 计算亮度之和
    result = np.sum(v_day)
    # 返回亮度的平均值
    return result / (height * width)

def crop_pic(contours, src):
    contours = np.array(contours, np.int32)
    mask = np.zeros(src.shape[:2], np.uint8)
    cv2.fillPoly(mask, [contours], (255, 255, 255))
    result = cv2.bitwise_and(src, src, mask=mask)
    return result

class SegPaste:
    def __init__(self, yaml_conf) -> None:
        self.learning_map = yaml_conf['learning_map']
        self.pandaset_dir = yaml_conf['pandaset_dir']
        self.seg_grabcut_dir = yaml_conf['seg_grabcut_dir']
        self.seg_seed_lidar_dir = yaml_conf['seg_seed_lidar_dir']
        self.lidar_pcd = open3d.geometry.PointCloud()
        self.get_all_seg_seed_jpg()

    def get_all_seg_seed_jpg(self):
        self.all_seg_seed_jpg = {}
        grabcut_mode_list = ['opencv', 'sam', 'removebg']
        for root, dir, files in os.walk(self.seg_grabcut_dir):
            for file in files:
                seg_seed_jpg_path = os.path.join(root, file)
                grabcut_mode, sub, idx, camera_name, label, ori_label, jpg_name = seg_seed_jpg_path.split('/')[-7:]
                grabcut_mode_idx = grabcut_mode_list.index(grabcut_mode)
                assert not os.path.exists(seg_seed_jpg_path.replace(grabcut_mode, grabcut_mode_list[grabcut_mode_idx - len(grabcut_mode_list) + 1]))
                assert not os.path.exists(seg_seed_jpg_path.replace(grabcut_mode, grabcut_mode_list[grabcut_mode_idx - len(grabcut_mode_list) + 2]))
                if label not in self.all_seg_seed_jpg:
                    self.all_seg_seed_jpg.update({label: []})
                self.all_seg_seed_jpg[label].append(seg_seed_jpg_path)

    def seg_paste(self, seg_seed_jpg_path, tar_paste_info):
        seg_seed_info = self.get_seed_jpg_info(seg_seed_jpg_path)
        projected_points2d, seg_seed_lidar_data_paste = self.spherical_coord_ego_lidar_paste(seg_seed_info, tar_paste_info)
        seed_mask_texture, seed_jpg, mid_offset, seed_mask_texture_mid = self.get_project_texture(projected_points2d, seg_seed_jpg_path)
        seg_seed_lidar_data_paste = self.repair_seg_lidar_data(seed_mask_texture, None, projected_points2d, seg_seed_lidar_data_paste, tar_paste_info, 0)
        tar_paste_info, seg_seed_lidar_data_pro, seg_mask_cover_idx = self.filter_overlap_pts(seg_seed_lidar_data_paste,
                                                                                        seed_mask_texture,
                                                                                        seg_seed_info,
                                                                                        tar_paste_info)
        seg_mask_reamin_pts = seed_mask_texture[seg_mask_cover_idx]
        seg_mask_cover_pts = seed_mask_texture[~seg_mask_cover_idx]
        tar_camera_texture = self.seed_texture(seed_jpg, seg_mask_reamin_pts, seg_mask_cover_pts, mid_offset, tar_paste_info['camera_data'], seed_mask_texture_mid)
        # cv2.imwrite('work_dirs/pandaset/jpg/seg_paste/texture_tree.jpg', tar_camera_texture)
        projected_points2d, __, _ = geometry.projection(lidar_points=tar_paste_info['lidar_data'][:,:3], 
                                                            camera_data=tar_paste_info['camera_data'],
                                                            camera_pose=tar_paste_info['calib_info']['camera_pose'],
                                                            camera_intrinsics=tar_paste_info['calib_info']['camera_intrinsics'],
                                                            filter_outliers=True)
        img = np.array(tar_paste_info['camera_data'])
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        for pts in seg_mask_reamin_pts:
            cv2.circle(img, (int(pts[0]), int(pts[1])), 1, (0,0,0), 4)
        for pts in projected_points2d:
            cv2.circle(img, (int(pts[0]), int(pts[1])), 1, (0,255,0), 4)
        # cv2.imwrite('work_dirs/pandaset/jpg/seg_paste/cover.jpg', img)
        # write_data(tar_paste_info['lidar_data'], 'work_dirs/pandaset/jpg/seg_paste', 'seg_paste', 6)
            
    
    def repair_seg_lidar_data(self, seed_mask_texture, seed_lidar_paste_match_a, seed_lidar_pro, seed_lidar_paste, tar_paste_info, recur_num):
        match_flag = np.ones(seed_lidar_pro.shape[0], dtype=bool)
        if seed_mask_texture.dtype != np.uint32:
            seed_mask_texture = seed_mask_texture.astype(np.uint32)
        if seed_lidar_pro.dtype != np.uint32:
            seed_lidar_pro = seed_lidar_pro.astype(np.uint32)
        seed_mask_texture_l = seed_mask_texture.tolist()
        seed_lidar_pro_l = seed_lidar_pro.tolist()
        for i in range(len(seed_lidar_pro_l)):
            pts = seed_lidar_pro_l[i]
            if pts in seed_mask_texture_l:continue
            match_flag[i] = False
        seed_lidar_paste_match = seed_lidar_paste[match_flag]
        if seed_lidar_paste_match_a is not None:
            seed_lidar_paste_match = np.concatenate((seed_lidar_paste_match, seed_lidar_paste_match_a), axis=0)
        seed_lidar_paste_mismatch = seed_lidar_paste[~match_flag]
        if recur_num >=3: return seed_lidar_paste_match
        elif recur_num == 0 or recur_num == 1:
            random_trans_dis = (np.random.rand(seed_lidar_paste_mismatch.shape[0], 6) - np.array([0.5, 0.5, 0.5, 0, 0, 0])) / (5-recur_num*2)
            seed_lidar_paste_mismatch_trans = seed_lidar_paste_mismatch[:, :3] + random_trans_dis[:, :3]
            seed_lidar_paste_mismatch_trans = np.concatenate((seed_lidar_paste_mismatch_trans, seed_lidar_paste_mismatch[:,3:]), axis=1)
        elif recur_num == 2:
            seed_lidar_paste_mismatch_trans = np.zeros(seed_lidar_paste_match.shape[0], dtype=bool)
            idx_list = [x for x in range(len(seed_lidar_paste_match))]
            random_idx_list = random.sample(idx_list, min(len(idx_list), len(seed_lidar_paste_mismatch)))
            for idx in random_idx_list:
                seed_lidar_paste_mismatch_trans[idx] = True
            lidar_paste_match_random_data = seed_lidar_paste_match[seed_lidar_paste_mismatch_trans]
            random_trans_dis = (np.random.rand(lidar_paste_match_random_data.shape[0], 6) - np.array([0.5, 0.5, 0.5, 0, 0, 0])) / 5
            seed_lidar_paste_mismatch_trans = lidar_paste_match_random_data[:, :3] + random_trans_dis[:, :3]
            seed_lidar_paste_mismatch_trans = np.concatenate((seed_lidar_paste_mismatch_trans, lidar_paste_match_random_data[:,3:]), axis=1)

        projected_points2d, _, __ = geometry.projection(lidar_points=seed_lidar_paste_mismatch_trans[:,:3], 
                                                        camera_data=tar_paste_info['camera_data'],
                                                        camera_pose=tar_paste_info['calib_info']['camera_pose'],
                                                        camera_intrinsics=tar_paste_info['calib_info']['camera_intrinsics'],
                                                        filter_outliers=True)
        seed_lidar_paste_match = self.repair_seg_lidar_data(seed_mask_texture, 
                                                            seed_lidar_paste_match,
                                                            projected_points2d, 
                                                            seed_lidar_paste_mismatch_trans,
                                                            tar_paste_info,
                                                            recur_num+1)
        return seed_lidar_paste_match
    
    def spherical_coord_ego_lidar_paste(self, seg_seed_info, tar_paste_info):
        """
        在ego自车坐标系转球面坐标系下平移旋转lidar点进行lidar paste数据增强
        """
        spherical_lidar_data = lidar2ego2spherical(seg_seed_info['seg_seed_lidar_data'][:, :3], seg_seed_info['lidar_pose'])
        self.lidar_pcd.points = open3d.utility.Vector3dVector(spherical_lidar_data)
        translate_dis = (65, 0, -5)
        self.lidar_pcd.translate(translate_dis, relative=True)
        ego2lidar_data = spherical2ego2lidar(np.array(self.lidar_pcd.points), tar_paste_info['calib_info']['lidar_pose'])
        projected_points2d, camera_points_3d, _ = geometry.projection(lidar_points=ego2lidar_data, 
                                                            camera_data=tar_paste_info['camera_data'],
                                                            camera_pose=tar_paste_info['calib_info']['camera_pose'],
                                                            camera_intrinsics=tar_paste_info['calib_info']['camera_intrinsics'],
                                                            filter_outliers=True)
        ego2lidar_data = np.concatenate((ego2lidar_data, seg_seed_info['seg_seed_lidar_data'][:, 3:]), axis=1)
        return projected_points2d, ego2lidar_data

    def get_project_texture(self, project_point2d, seg_seed_jpg_path):
        project_point2d = project_point2d.astype(np.uint32)
        h = np.max(project_point2d[:,1]) - np.min(project_point2d[:,1])
        w = np.max(project_point2d[:,0]) - np.min(project_point2d[:,0])
        project_point2d_mid_y = (np.max(project_point2d[:,1])+np.min(project_point2d[:,1])) // 2
        project_point2d_mid_x = (np.max(project_point2d[:,0])+np.min(project_point2d[:,0])) // 2
        seed_mask_jpg = cv2.imread(seg_seed_jpg_path, 0)
        seed_jpg = cv2.imread(seg_seed_jpg_path, 1)
        ori_h, ori_w = seed_mask_jpg.shape[0] - 34, seed_mask_jpg.shape[1] - 8
        seed_mask_jpg = cv2.resize(seed_mask_jpg, (0,0), fx=w/ori_w, fy=h/ori_h)
        seed_jpg = cv2.resize(seed_jpg, (0,0), fx=w/ori_w, fy=h/ori_h)
        left, right = int(min(project_point2d[:,0]) - 4 * w / ori_w), int(max(project_point2d[:,0]) + 4 * w / ori_w)
        low, high = int(min(project_point2d[:,1]) - 30 * h / ori_h), int(max(project_point2d[:,1]) + 4 * h / ori_h)
        mid_x, mid_y = (left+right) // 2, (low+high) // 2
        seed_mask_jpg_0 = np.where(seed_mask_jpg != 0)
        seed_mask = np.squeeze(np.dstack((seed_mask_jpg_0[1], seed_mask_jpg_0[0])))
        seed_mask_jpg_mid_x = seed_mask_jpg.shape[1] // 2
        seed_mask_jpg_mid_y = seed_mask_jpg.shape[0] // 2
        mid_offset = np.array((mid_x-seed_mask_jpg_mid_x, mid_y-seed_mask_jpg_mid_y))
        seed_mask_texture = seed_mask + mid_offset
        return seed_mask_texture, seed_jpg, mid_offset, (project_point2d_mid_y, project_point2d_mid_x)


    def filter_overlap_pts(self, seg_seed_lidar_data, seed_mask_texture, seg_seed_info, tar_paste_info):
        projected_points2d, _, inner_indices = geometry.projection(lidar_points=tar_paste_info['lidar_data'][:,:3], 
                                                                camera_data=tar_paste_info['camera_data'],
                                                                camera_pose=tar_paste_info['calib_info']['camera_pose'],
                                                                camera_intrinsics=tar_paste_info['calib_info']['camera_intrinsics'],
                                                                filter_outliers=True)
        tar_camera_lidar_data = tar_paste_info['lidar_data'][inner_indices]
        tar_camera_semseg = tar_paste_info['semseg'][inner_indices]
        tar_camera_lidar2spherical = lidar2ego2spherical(tar_camera_lidar_data[:,:3], tar_paste_info['calib_info']['lidar_pose'])
        seg_seed_lidar2spherical_data = lidar2ego2spherical(seg_seed_lidar_data[:,:3], seg_seed_info['lidar_pose'])
        seg_seed_lidar2spherical_data_pro, _, __ = geometry.projection(lidar_points=seg_seed_lidar_data[:,:3],
                                                                    camera_data=seg_seed_info['camera_data'],
                                                                    camera_pose=seg_seed_info['camera_pose'],
                                                                    camera_intrinsics=seg_seed_info['camera_intrinsics'],
                                                                    filter_outliers=True)
        front_overlap_data, tar_cover_idx = self.get_overlap_pts(projected_points2d,
                                                            seed_mask_texture,
                                                            tar_camera_semseg,
                                                            tar_camera_lidar2spherical,
                                                            seg_seed_lidar2spherical_data,
                                                            seg_seed_lidar2spherical_data_pro)
        seg_cover_idx, seg_mask_cover_idx = self.filter_front_mask_overlap(front_overlap_data,
                                                                            seg_seed_lidar2spherical_data_pro,
                                                                            seed_mask_texture,
                                                                            tar_camera_semseg,
                                                                            tar_camera_lidar2spherical,
                                                                            tar_paste_info)
        seg_seed_lidar_data = seg_seed_lidar_data[seg_cover_idx]
        seg_seed_lidar2spherical_data_pro = seg_seed_lidar2spherical_data_pro[seg_cover_idx]
        tar_camera_lidar_data = tar_camera_lidar_data[tar_cover_idx]
        tar_camera_semseg = tar_camera_semseg[tar_cover_idx]
        seg_seed_semseg_data = (np.ones(seg_seed_lidar_data.shape[0]) * 5).reshape(-1,1) #label
        tar_paste_info['lidar_data'] = np.delete(tar_paste_info['lidar_data'], inner_indices, 0)
        tar_paste_info['semseg'] = np.delete(tar_paste_info['semseg'], inner_indices, 0)
        tar_paste_info['lidar_data'] = np.concatenate((tar_paste_info['lidar_data'], tar_camera_lidar_data, seg_seed_lidar_data), axis=0)
        tar_paste_info['semseg'] = np.concatenate((tar_paste_info['semseg'], tar_camera_semseg, seg_seed_semseg_data), axis=0)
        return tar_paste_info, seg_seed_lidar2spherical_data_pro, seg_mask_cover_idx
    
    def seed_texture(self, seed_jpg, seg_mask_reamin_pts, seg_mask_cover_pts, mid_offset, tar_camera_data, seed_mask_texture_mid):
        seg_mask_cover_pts = seg_mask_cover_pts - mid_offset
        for pts in seg_mask_cover_pts:
            seed_jpg[pts[1], pts[0]] = (0,0,0)
        seed_jpg_gray = cv2.cvtColor(seed_jpg, cv2.COLOR_RGB2GRAY)
        _, seed_mask = cv2.threshold(seed_jpg_gray, 10, 255, cv2.THRESH_BINARY)
        tar_camera_img = np.array(tar_camera_data)
        tar_camera_img = cv2.cvtColor(tar_camera_img,cv2.COLOR_RGB2BGR)
        contours = [[seed_mask_texture_mid[1]-int(seed_jpg.shape[1]/2), seed_mask_texture_mid[0]-int(seed_jpg.shape[0]/2)],
                    [seed_mask_texture_mid[1]-int(seed_jpg.shape[1]/2), seed_mask_texture_mid[0]+int(seed_jpg.shape[0]/2)],
                    [seed_mask_texture_mid[1]+int(seed_jpg.shape[1]/2), seed_mask_texture_mid[0]+int(seed_jpg.shape[0]/2)],
                    [seed_mask_texture_mid[1]+int(seed_jpg.shape[1]/2), seed_mask_texture_mid[0]-int(seed_jpg.shape[0]/2)]]
        img_c = crop_pic(contours, tar_camera_img)
        brightness = average_brightness(img_c)
        img = self.copy_paste(seed_jpg, seg_mask_reamin_pts, tar_camera_img, seed_mask, seed_mask_texture_mid, brightness)
        return img

    def copy_paste(self, img_src, seg_mask_reamin_pts, img_main, mask_src, center, brightness):
        mid_x, mid_y = center[0], center[1]
        left = mid_x - int(img_src.shape[0] // 2)
        right = mid_y - int(img_src.shape[1] // 2)
        if len(img_main.shape) == 3:
            h, w, c = img_main.shape
        elif len(img_main.shape) == 2:
            h, w = img_main.shape
        mask = np.asarray(mask_src, dtype=np.uint8)
        sub_img01 = cv2.add(img_src, np.zeros(np.shape(img_src), dtype=np.uint8), mask=mask)
        # mask_02 = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_02 = np.asarray(mask, dtype=np.uint8)
        img_main_c = img_main[left:(left)+img_src.shape[0], right:(right+img_src.shape[1])]
        sub_img02 = cv2.add(img_main_c, np.zeros(np.shape(img_main_c), dtype=np.uint8),
                            mask=mask_02)
        img_main_c = img_main_c - sub_img02 + sub_img01
        img_main[left:(left)+img_src.shape[0], right:(right+img_src.shape[1])] = img_main_c

        img_zero = np.zeros(img_main.shape, dtype=np.uint8)
        for pts in seg_mask_reamin_pts:
            img_zero[int(pts[1]), int(pts[0])] = 255
        if brightness <= 1:
            alpha, beta = 0.4, 0.4
        elif 1 < brightness <= 1.5:
            alpha, beta = 0.3, 0.3
        elif 1.5 < brightness <= 2.5:
            alpha, beta = 0.25, 0.25
        elif 2.5 < brightness <= 3:
            alpha, beta = 0.15, 0.15
        else:
            alpha, beta = 0.1, 0.1
        img_ill = cv2.illuminationChange(img_main, img_zero, alpha=alpha, beta=beta)
        return img_ill
    
    def get_overlap_pts(self, projected_points2d, seed_mask_texture, tar_camera_semseg, tar_camera_lidar2spherical, seg_seed_lidar2spherical_data, seg_seed_lidar2spherical_data_pro):
        front_overlap_data = []
        projected_points2d_l = projected_points2d.astype(np.uint32).tolist()
        seed_mask_texture_l = seed_mask_texture.tolist()
        tar_cover_idx = np.ones(tar_camera_lidar2spherical.shape[0], dtype=bool)
        for i in range(len(projected_points2d_l)):
            pts = projected_points2d_l[i]
            if pts in seed_mask_texture_l:
                tar_camera_lidar2spherical_idx = tar_camera_lidar2spherical[i]
                tar_camera_semseg_idx = tar_camera_semseg[i]
                seg_seed_close_data = self.get_close_pts(pts, seg_seed_lidar2spherical_data, seg_seed_lidar2spherical_data_pro)
                if tar_camera_lidar2spherical_idx[2] >= seg_seed_close_data[2]:
                    tar_cover_idx[i] = False
                else:
                    front_overlap_data.append(np.concatenate((tar_camera_lidar2spherical_idx, tar_camera_semseg_idx)))
        return front_overlap_data, tar_cover_idx

    def get_close_pts(self, pt, spherical_data, spherical_data_pro):
        pt_n = np.array([pt] * spherical_data_pro.shape[0])
        data = spherical_data_pro - pt_n
        dis = np.sqrt(np.sum(np.square(data), axis=1))
        min_dis = np.min(dis)
        min_dis_idx = np.where(dis == min_dis)[0][0]
        return spherical_data[min_dis_idx]
    
    def filter_front_mask_overlap(self, front_overlap_data, seg_seed_lidar2spherical_data_pro, seed_mask_texture, tar_camera_semseg, tar_camera_lidar2spherical, tar_paste_info):
        seg_cover_idx = np.ones(seg_seed_lidar2spherical_data_pro.shape[0], dtype=bool)
        seg_mask_cover_idx = np.ones(seed_mask_texture.shape[0], dtype=bool)
        if len(front_overlap_data) > 1:
            front_overlap_data = np.array(front_overlap_data)
            labels = np.unique(front_overlap_data[:,-1])
            for label in labels:
                front_overlap_label_data = front_overlap_data[np.where(front_overlap_data[:,-1]==label)]
                label_idx = np.ones(tar_camera_semseg.shape[0], dtype=bool)
            #     label_idx = np.logical_and(label_idx, tar_camera_semseg[:, 0]==label)
            #     tar_camera_label_lidar2spherical = tar_camera_lidar2spherical[label_idx]
            #     tar_camera_label_spherical2lidar = lidar2ego2spherical(tar_camera_label_lidar2spherical, tar_paste_info['calib_info']['lidar_pose'])
            #     clustering = DBSCAN(eps=1.5, min_samples=15).fit(tar_camera_label_spherical2lidar)
            #     cluster_labels = clustering.labels_
            #     for i in range(max(cluster_labels)+1):
            #         cluster_label_idx = np.ones(tar_camera_label_spherical2lidar.shape[0], dtype=bool)
            #         cluster_label_idx = np.logical_and(cluster_label_idx, np.array(cluster_labels)==i)
            #         cluster_label_data = tar_camera_label_spherical2lidar[cluster_label_idx]
            #         if front_overlap_label_data[0,:3].tolist() not in cluster_label_data.tolist(): continue
            #         label_project_data, _, __ = geometry.projection(cluster_label_data,
            #                                                         camera_data=tar_paste_info['camera_data'],
            #                                                         camera_pose=tar_paste_info['calib_info']['camera_pose'],
            #                                                         camera_intrinsics=tar_paste_info['calib_info']['camera_intrinsics'],
            #                                                         filter_outliers=True)
            label_project_data = read_data('work_dirs/pandaset/jpg/car/project_point_data/label_5_project_point_data', 2).tolist()
            alpha_shape=alphashape.alphashape(label_project_data,0)
            alpha_shape = list(alpha_shape.exterior.coords)
            polygon = Polygon(alpha_shape)
            for i in range(len(seg_seed_lidar2spherical_data_pro)):
                pts = seg_seed_lidar2spherical_data_pro[i].tolist()
                point = Point(pts[0], pts[1])
                if polygon.contains(point):
                    seg_cover_idx[i] = False
            for i in range(len(seed_mask_texture)):
                pts = seed_mask_texture[i].tolist()
                point = Point(pts[0], pts[1])
                if polygon.contains(point):
                    seg_mask_cover_idx[i] = False
        return seg_cover_idx, seg_mask_cover_idx
    
    def filter_mask_overlap(self, seg_mask_cover_idx, seed_mask_texture):
        seg_mask_cover = np.ones(seed_mask_texture.shape[0], dtype=bool)
        seg_mask_cover_pts = seed_mask_texture[seg_mask_cover_idx].tolist()
        alpha_shape=alphashape.alphashape(seg_mask_cover_pts,0)
        alpha_shape = list(alpha_shape.exterior.coords)
        polygon = Polygon(alpha_shape)
        for i in range(seed_mask_texture.shape[0]):
            pts = seed_mask_texture[i].tolist()
            point = Point(pts[0], pts[1])
            if polygon.contains(point):
                seg_mask_cover[i] = False
        return seed_mask_texture[seg_mask_cover]

    def get_low_lidar_center(self, lidar_data, length):
        lidar_data_sort = np.sort(lidar_data, axis=0)
        length = min(lidar_data_sort.shape[0], length)
        ext_lidar_data = lidar_data_sort[:length, :]
        low_center = np.mean(ext_lidar_data, axis=0)
        low_center[2] = np.min(ext_lidar_data[:,2])
        return low_center

    def get_seed_jpg_info(self, seed_jpg_path):
        seed_jpg_path_split = seed_jpg_path.split('/')
        grabcut_mode, sub, idx, camera_name, label, ori_label, jpg_name = seed_jpg_path_split[-7:]
        camera_data = Image.open(os.path.join(self.pandaset_dir, sub, 'camera', camera_name, idx + '.jpg'))
        camera_pose = json.load(open(os.path.join(self.pandaset_dir, sub, 'camera', camera_name, 'poses.json')))[int(idx)]
        camera_intrinsics = json.load(open(os.path.join(self.pandaset_dir, sub, 'camera', camera_name, 'intrinsics.json')))
        camera_intrinsics = Intrinsics(fx=camera_intrinsics['fx'],
                                        fy=camera_intrinsics['fy'],
                                        cx=camera_intrinsics['cx'],
                                        cy=camera_intrinsics['cy'])
        lidar_pose = json.load(open(os.path.join(self.pandaset_dir, sub, 'lidar', 'poses.json')))[int(idx)]
        seed_lidar_data_path = os.path.join(self.seg_seed_lidar_dir, sub, idx, camera_name, label, ori_label, jpg_name.split('.jpg')[0])
        seg_seed_lidar_data = read_data(seed_lidar_data_path, 6)
        projected_points2d, camera_points_3d, _ = geometry.projection(lidar_points=seg_seed_lidar_data[:, :3], 
                                                            camera_data=camera_data,
                                                            camera_pose=camera_pose,
                                                            camera_intrinsics=camera_intrinsics,
                                                            filter_outliers=True)
        seg_seed_info = {
            'camera_data': camera_data,
            'camera_pose': camera_pose,
            'camera_intrinsics': camera_intrinsics,
            'lidar_pose': lidar_pose,
            'seg_seed_lidar_data': seg_seed_lidar_data,
            'label': label,
            'ori_label': ori_label,
            'lidar_projected_points2d': projected_points2d
        }
        return seg_seed_info


if __name__ == '__main__':
    pass