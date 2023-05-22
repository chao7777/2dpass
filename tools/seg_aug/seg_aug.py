from copyreg import add_extension
import cv2
import open3d
from tools.pandaset import DataSet
from tools.pandaset import geometry
from tqdm import tqdm
import numpy as np
import random
from sklearn.cluster import DBSCAN
import os
from shapely.geometry import Polygon,Point
import alphashape
from get_seed import write_data, MultiProcess, read_data

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

class SegAug:
    def __init__(self, seg_seq, tar_seq) -> None:
        self.seg_seq = seg_seq
        self.tar_seq = tar_seq
        self.lidar_pcd = open3d.geometry.PointCloud()

    def seg_paste(self, seg_seed_lidar_data_path, tar_camera_name, tar_seq_idx):
        seg_seed_lidar_data = np.array(read_data(seg_seed_lidar_data_path, 6))
        seg_seed_lidar_data_info = self.get_lidar_info(seg_seed_lidar_data_path)
        tar_camera = self.tar_seq.camera[tar_camera_name]
        tar_lidar = self.tar_seq.lidar
        points3d_lidar_xyz = tar_lidar.data[tar_seq_idx].to_numpy()
        tar_semseg = self.tar_seq.semseg[tar_seq_idx].to_numpy()
        # points3d_lidar_xyz = np.array(read_data('work_dirs/pandaset/jpg/car/clust_label_data/label_5_lidar_data'))
        # points3d_lidar_xyz_ = np.array(read_data('work_dirs/pandaset/jpg/building/clust_label_data/label_11_lidar_data'))
        # points3d_lidar_xyz = np.concatenate((points3d_lidar_xyz, points3d_lidar_xyz_), axis=0)
        # tar_semseg = np.ones(points3d_lidar_xyz.shape[0]).reshape(-1,1)
        projected_points2d, seg_seed_lidar_data_paste = self.spherical_coord_ego_lidar_paste(seg_seed_lidar_data, 
                                                                                             seg_seed_lidar_data_info,
                                                                                             tar_lidar,
                                                                                             tar_camera, 
                                                                                             tar_seq_idx)
        seed_mask_texture, seed_jpg, mid_offset, seed_mask_texture_mid = self.get_project_texture(projected_points2d, seg_seed_lidar_data_path)
        seg_seed_lidar_data_paste = self.repair_seg_lidar_data(seed_mask_texture, None, projected_points2d, seg_seed_lidar_data_paste, tar_camera, tar_seq_idx,0)
        points3d_lidar_xyz, tar_semseg, seg_seed_lidar_data_pro, seg_mask_cover_idx = self.filter_overlap_pts(seg_seed_lidar_data_paste,
                                                                                        seed_mask_texture,
                                                                                        seg_seed_lidar_data_info,
                                                                                        points3d_lidar_xyz,
                                                                                        tar_semseg,
                                                                                        tar_lidar,
                                                                                        tar_camera,
                                                                                        tar_seq_idx)
        seg_mask_reamin_pts = seed_mask_texture[seg_mask_cover_idx]
        seg_mask_cover_pts = seed_mask_texture[~seg_mask_cover_idx]
        tar_camera_texture = self.seed_texture(seed_jpg, seg_mask_reamin_pts, seg_mask_cover_pts, mid_offset, tar_camera[tar_seq_idx], seed_mask_texture_mid)
        cv2.imwrite('work_dirs/pandaset/jpg/seg_paste/texture_tree.jpg', tar_camera_texture)
        projected_points2d, __, _ = geometry.projection(lidar_points=points3d_lidar_xyz[:,:3], 
                                                            camera_data=tar_camera[tar_seq_idx],
                                                            camera_pose=tar_camera.poses[tar_seq_idx],
                                                            camera_intrinsics=tar_camera.intrinsics,
                                                            filter_outliers=True)
        img = np.array(tar_camera[tar_seq_idx])
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        for pts in seg_mask_reamin_pts:
            cv2.circle(img, (int(pts[0]), int(pts[1])), 1, (0,0,0), 4)
        for pts in projected_points2d:
            cv2.circle(img, (int(pts[0]), int(pts[1])), 1, (0,255,0), 4)
        cv2.imwrite('work_dirs/pandaset/jpg/seg_paste/cover.jpg', img)
        # write_data(points3d_lidar_xyz, 'work_dirs/pandaset/jpg/seg_paste', 'seg_paste', 6)
    
    def repair_seg_lidar_data(self, seed_mask_texture, seed_lidar_paste_match_a, seed_lidar_pro, seed_lidar_paste, tar_camera, tar_seq_idx, num):
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
        if num >=3: return seed_lidar_paste_match
        elif num == 0 or num == 1:
            random_trans_dis = (np.random.rand(seed_lidar_paste_mismatch.shape[0], 6) - np.array([0.5, 0.5, 0.5, 0, 0, 0])) / (5-num*2)
            seed_lidar_paste_mismatch_trans = seed_lidar_paste_mismatch[:, :3] + random_trans_dis[:, :3]
            seed_lidar_paste_mismatch_trans = np.concatenate((seed_lidar_paste_mismatch_trans, seed_lidar_paste_mismatch[:,3:]), axis=1)
        elif num == 2:
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
                                                        camera_data=tar_camera[tar_seq_idx],
                                                        camera_pose=tar_camera.poses[tar_seq_idx],
                                                        camera_intrinsics=tar_camera.intrinsics,
                                                        filter_outliers=True)
        seed_lidar_paste_match = self.repair_seg_lidar_data(seed_mask_texture, 
                                                            seed_lidar_paste_match,
                                                            projected_points2d, 
                                                            seed_lidar_paste_mismatch_trans,
                                                            tar_camera,
                                                            tar_seq_idx,
                                                            num+1)
        return seed_lidar_paste_match
    
    def spherical_coord_ego_lidar_paste(self, seg_seed_lidar_data, seg_seed_lidar_data_info, tar_lidar, tar_camera, tar_seq_idx):
        """
        在ego自车坐标系转球面坐标系下平移旋转lidar点进行lidar paste数据增强
        """
        spherical_lidar_data = self.lidar2ego2spherical(seg_seed_lidar_data[:, :3], self.seg_seq.lidar.poses[seg_seed_lidar_data_info['seq_idx']])
        self.lidar_pcd.points = open3d.utility.Vector3dVector(spherical_lidar_data)
        translate_dis = (65, 0, -5)
        self.lidar_pcd.translate(translate_dis, relative=True)
        ego2lidar_data = self.spherical2ego2lidar(np.array(self.lidar_pcd.points), tar_lidar.poses[tar_seq_idx])
        write_data(ego2lidar_data, 'work_dirs/pandaset/jpg/translate/spherical/ego/pcd', '{}_0_{}'.format(translate_dis[0], translate_dis[2]), 3)
        projected_points2d, camera_points_3d, _ = geometry.projection(lidar_points=ego2lidar_data, 
                                                            camera_data=tar_camera[tar_seq_idx],
                                                            camera_pose=tar_camera.poses[tar_seq_idx],
                                                            camera_intrinsics=tar_camera.intrinsics,
                                                            filter_outliers=True)
        img = np.array(tar_camera[tar_seq_idx])
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        for pts in projected_points2d:
            cv2.circle(img, (int(pts[0]), int(pts[1])), 1, (0,255,0), 4)
        cv2.imwrite('work_dirs/pandaset/jpg/translate/spherical/ego/gt_aug_{}_0_{}.jpg'.format(translate_dis[0], translate_dis[2]), img)
        ego2lidar_data = np.concatenate((ego2lidar_data, seg_seed_lidar_data[:,3:]), axis=1)
        return projected_points2d, ego2lidar_data

    def get_project_texture(self, project_point2d, ori_lidar_data_path):
        project_point2d = project_point2d.astype(np.uint32)
        h = np.max(project_point2d[:,1])-np.min(project_point2d[:,1])
        w = np.max(project_point2d[:,0])-np.min(project_point2d[:,0])
        project_point2d_mid_y = (np.max(project_point2d[:,1])+np.min(project_point2d[:,1])) // 2
        project_point2d_mid_x = (np.max(project_point2d[:,0])+np.min(project_point2d[:,0])) // 2
        seed_mask_jpg = self.read_pic(ori_lidar_data_path, color_t=0)
        seed_jpg = self.read_pic(ori_lidar_data_path, color_t=1)
        ori_h, ori_w = seed_mask_jpg.shape[0]-27, seed_mask_jpg.shape[1]-4
        seed_mask_jpg = cv2.resize(seed_mask_jpg, (0,0), fx=w/ori_w, fy=h/ori_h)
        seed_jpg = cv2.resize(seed_jpg, (0,0), fx=w/ori_w, fy=h/ori_h)
        left, right = int(min(project_point2d[:,0]) - 2*w/ori_w), int(max(project_point2d[:,0]) + 2*w/ori_w)
        low, high = int(min(project_point2d[:,1]) - 25*h/ori_h), int(max(project_point2d[:,1]) + 2*h/ori_h)
        mid_x, mid_y = (left+right) // 2, (low+high) // 2
        seed_mask_jpg_0 = np.where(seed_mask_jpg != 0)
        seed_mask = np.squeeze(np.dstack((seed_mask_jpg_0[1], seed_mask_jpg_0[0])))
        seed_mask_jpg_mid_x = seed_mask_jpg.shape[1] // 2
        seed_mask_jpg_mid_y = seed_mask_jpg.shape[0] // 2
        mid_offset = np.array((mid_x-seed_mask_jpg_mid_x, mid_y-seed_mask_jpg_mid_y))
        seed_mask_texture = seed_mask + mid_offset
        return seed_mask_texture, seed_jpg, mid_offset, (project_point2d_mid_y, project_point2d_mid_x)


    def filter_overlap_pts(self, seg_seed_lidar_data, seed_mask_texture, seg_seed_lidar_data_info, points3d_lidar_xyz, tar_semseg, tar_lidar, tar_camera, tar_seq_idx):
        projected_points2d, _, inner_indices = geometry.projection(lidar_points=points3d_lidar_xyz[:,:3], 
                                                                camera_data=tar_camera[tar_seq_idx],
                                                                camera_pose=tar_camera.poses[tar_seq_idx],
                                                                camera_intrinsics=tar_camera.intrinsics,
                                                                filter_outliers=True)
        tar_camera_lidar_data = points3d_lidar_xyz[inner_indices]
        tar_camera_semseg = tar_semseg[inner_indices]
        tar_camera_lidar2spherical = self.lidar2ego2spherical(tar_camera_lidar_data[:,:3], tar_lidar.poses[tar_seq_idx])
        seg_seed_lidar2spherical_data = self.lidar2ego2spherical(seg_seed_lidar_data[:,:3], self.seg_seq.lidar.poses[seg_seed_lidar_data_info['seq_idx']])
        seg_seed_camera = self.seg_seq.camera[seg_seed_lidar_data_info['camera_name']]
        seg_seed_lidar2spherical_data_pro, _, __ = geometry.projection(lidar_points=seg_seed_lidar_data[:,:3],
                                                                    camera_data=seg_seed_camera[seg_seed_lidar_data_info['seq_idx']],
                                                                    camera_pose=seg_seed_camera.poses[seg_seed_lidar_data_info['seq_idx']],
                                                                    camera_intrinsics=seg_seed_camera.intrinsics,
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
                                                                            tar_lidar.poses[tar_seq_idx],
                                                                            tar_camera,
                                                                            tar_seq_idx)
        seg_seed_lidar_data = seg_seed_lidar_data[seg_cover_idx]
        seg_seed_lidar2spherical_data_pro = seg_seed_lidar2spherical_data_pro[seg_cover_idx]
        tar_camera_lidar_data = tar_camera_lidar_data[tar_cover_idx]
        tar_camera_semseg = tar_camera_semseg[tar_cover_idx]
        seg_seed_semseg_data = (np.ones(seg_seed_lidar_data.shape[0]) * 5).reshape(-1,1) #label
        points3d_lidar_xyz = np.delete(points3d_lidar_xyz, inner_indices, 0)
        tar_semseg = np.delete(tar_semseg, inner_indices, 0)
        points3d_lidar_xyz = np.concatenate((points3d_lidar_xyz, tar_camera_lidar_data, seg_seed_lidar_data), axis=0)
        # points3d_lidar_xyz = np.concatenate((points3d_lidar_xyz, tar_camera_lidar_data), axis=0)
        tar_semseg = np.concatenate((tar_semseg, tar_camera_semseg, seg_seed_semseg_data), axis=0)
        return points3d_lidar_xyz, tar_semseg, seg_seed_lidar2spherical_data_pro, seg_mask_cover_idx
    
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
    
    def read_pic(self, ori_lidar_data_path, color_t=1):
        if os.path.exists(ori_lidar_data_path + '.jpg'):
            seed_jpg = cv2.imread(ori_lidar_data_path + '.jpg', color_t)
        elif os.path.exists(ori_lidar_data_path + '.png'):
            seed_jpg = cv2.imread(ori_lidar_data_path + '.png', color_t)
        else:
            raise 'seed jpg path {} not found'.format(ori_lidar_data_path)
        return seed_jpg
    
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
    
    def filter_front_mask_overlap(self, front_overlap_data, seg_seed_lidar2spherical_data_pro, seed_mask_texture, tar_camera_semseg, tar_camera_lidar2spherical, tar_lidar_pose, tar_camera, tar_seq_idx):
        seg_cover_idx = np.ones(seg_seed_lidar2spherical_data_pro.shape[0], dtype=bool)
        seg_mask_cover_idx = np.ones(seed_mask_texture.shape[0], dtype=bool)
        if len(front_overlap_data) > 1:
            front_overlap_data = np.array(front_overlap_data)
            labels = np.unique(front_overlap_data[:,-1])
            for label in labels:
                front_overlap_label_data = front_overlap_data[np.where(front_overlap_data[:,-1]==label)]
                label_idx = np.ones(tar_camera_semseg.shape[0], dtype=bool)
                # label_idx = np.logical_and(label_idx, tar_camera_semseg==label)
                # tar_camera_label_lidar2spherical = tar_camera_lidar2spherical[label_idx]
                # tar_camera_label_spherical2lidar = self.lidar2ego2spherical(tar_camera_label_lidar2spherical, tar_lidar_pose)
                # clustering = DBSCAN(eps=1.5, min_samples=15).fit(tar_camera_label_spherical2lidar)
                # cluster_labels = clustering.labels_
                # for i in range(max(cluster_labels)+1):
                #     cluster_label_idx = np.ones(tar_camera_label_spherical2lidar.shape[0], dtype=bool)
                #     cluster_label_idx = np.logical_and(cluster_label_idx, np.array(cluster_labels)==i)
                #     cluster_label_data = tar_camera_label_spherical2lidar[cluster_label_idx]
                #     if front_overlap_label_data[0,:3].tolist() not in cluster_label_data.tolist(): continue
                #     label_project_data, _, __ = geometry.projection(cluster_label_data,
                #                                                     camera_data=tar_camera[tar_seq_idx],
                #                                                     camera_pose=tar_camera.poses[tar_seq_idx],
                #                                                     camera_intrinsics=tar_camera.intrinsics,
                #                                                     filter_outliers=True)
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

    def lidar2ego2spherical(self, lidar_data, lidar_pose):
        lidar2ego_data = geometry.lidar_points_to_ego(lidar_data, lidar_pose)
        spherical_lidar_data = geometry.world2spherical(lidar2ego_data)
        return spherical_lidar_data

    def spherical2ego2lidar(self, spherical_lidar_data, lidar_pose):
        world_lidar_data = geometry.spherical2world(spherical_lidar_data)
        ego2lidar_data = geometry.ego_to_lidar_point(world_lidar_data, lidar_pose)
        return ego2lidar_data

    def get_low_lidar_center(self, lidar_data, length):
        lidar_data_sort = np.sort(lidar_data, axis=0)
        length = min(lidar_data_sort.shape[0], length)
        ext_lidar_data = lidar_data_sort[:length, :]
        low_center = np.mean(ext_lidar_data, axis=0)
        low_center[2] = np.min(ext_lidar_data[:,2])
        return low_center

    def get_lidar_info(self, lidar_data_path):
        split_path = lidar_data_path.split('/')
        label = split_path[-1].split('_')[1]
        cluster_label = split_path[-1].split('_')[2]
        lidar_data_info = {
            'label':label,
            'cluster_label':cluster_label,
            'camera_name':split_path[-2],
            'seq_idx':int(split_path[-3]),
            'sub_dir':split_path[-4]
        }
        return lidar_data_info

    def world_coord_lidar_paste(self, ori_lidar_data_path, tar_camera_name, tar_seq_idx):
        """
        在世界坐标系下平移旋转lidar点进行lidar paste数据增强
        """
        ori_lidar_data = np.array(read_data(ori_lidar_data_path, 6))
        ori_lidar_data_info = self.get_lidar_info(ori_lidar_data_path)
        lidar2ego_data = geometry.lidar_points_to_ego(ori_lidar_data[:, :3], self.seg_seq.lidar.poses[ori_lidar_data_info['seq_idx']])
        low_center = self.get_low_lidar_center(lidar2ego_data, 10)
        tar_lidar = self.tar_seq.lidar
        tar_lidar_data = tar_lidar[tar_seq_idx].to_numpy()[:, :3]
        tar_camera = self.tar_seq.camera[tar_camera_name]
        # 暂定迁移后的中心为（8.5，-13），z不变
        tar_low_center = np.array([5.5, -13, low_center[2]])
        translate_dis = tar_low_center - low_center
        self.lidar_pcd.points = open3d.utility.Vector3dVector(lidar2ego_data)
        self.lidar_pcd.translate(translate_dis, relative=True)
        lidar_center = self.lidar_pcd.get_center()
        R = self.lidar_pcd.get_rotation_matrix_from_xyz(rotation = [0, 0, np.radians(60)])
        self.lidar_pcd.rotate(R=R, center=lidar_center)
        write_data(np.array(self.lidar_pcd.points), 'work_dirs/pandaset/jpg/translate/world/pcd', '0_0_60', 3)
        ego2lidar_data = geometry.ego_to_lidar_point(np.array(self.lidar_pcd.points), tar_lidar.poses[tar_seq_idx])
        projected_points2d, camera_points_3d, _ = geometry.projection(lidar_points=ego2lidar_data, 
                                                            camera_data=tar_camera[tar_seq_idx],
                                                            camera_pose=tar_camera.poses[tar_seq_idx],
                                                            camera_intrinsics=tar_camera.intrinsics,
                                                            filter_outliers=True)
        img = np.array(tar_camera[tar_seq_idx])
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        for pts in projected_points2d:
            cv2.circle(img, (int(pts[0]), int(pts[1])), 1, (0,255,0), 4)
        cv2.imwrite('work_dirs/pandaset/jpg/translate/world/gt_aug_0_0_60.jpg', img)

    def spherical_coord_world_lidar_paste(self, ori_lidar_data_path, tar_camera_name, tar_seq_idx):
        """
        在世界坐标系转球面坐标系下平移旋转lidar点进行lidar paste数据增强
        """
        ori_lidar_data = np.array(read_data(ori_lidar_data_path, 6))
        tar_camera = self.tar_seq.camera[tar_camera_name]
        spherical_lidar_data = geometry.world2spherical(ori_lidar_data[:, :3])
        self.lidar_pcd.points = open3d.utility.Vector3dVector(spherical_lidar_data)
        translate_dis = (50, 0, 0)
        self.lidar_pcd.translate(translate_dis, relative=True)
        world_lidar_data = geometry.spherical2world(np.array(self.lidar_pcd.points))
        write_data(world_lidar_data, 'work_dirs/pandaset/jpg/translate/spherical/world/pcd', '{}_0_{}'.format(translate_dis[0], translate_dis[2]), 3)
        projected_points2d, camera_points_3d, _ = geometry.projection(lidar_points=world_lidar_data, 
                                                            camera_data=tar_camera[tar_seq_idx],
                                                            camera_pose=tar_camera.poses[tar_seq_idx],
                                                            camera_intrinsics=tar_camera.intrinsics,
                                                            filter_outliers=True)
        img = np.array(tar_camera[tar_seq_idx])
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        for pts in projected_points2d:
            cv2.circle(img, (int(pts[0]), int(pts[1])), 1, (0,255,0), 4)
        cv2.imwrite('work_dirs/pandaset/jpg/translate/spherical/world/gt_aug_{}_0_{}.jpg'.format(translate_dis[0], translate_dis[2]), img)

def test():
    lidar_data_path = '/share/qi.chao/open_sor_data/pandaset/grabcut_jpg/053/00/back_camera/label_5_3_1678_465'
    dataset = DataSet("/share/qi.chao/open_sor_data/pandaset/pandaset_1")
    seq053 = dataset["053"]
    seq053.load()
    seg_aug = SegAug(seq053, seq053)
    seg_aug.seg_paste(lidar_data_path, 'back_camera', 0)


if __name__ == '__main__':
    test()