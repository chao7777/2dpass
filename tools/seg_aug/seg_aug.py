import cv2
import open3d
from tools.pandaset import DataSet
from tools.pandaset import geometry
from tqdm import tqdm
import numpy as np
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
    
    def spherical_coord_ego_lidar_paste(self, seg_seed_lidar_data_path, tar_camera_name, tar_seq_idx):
        """
        在ego自车坐标系转球面坐标系下平移旋转lidar点进行lidar paste数据增强
        """
        seg_seed_lidar_data = np.array(read_data(seg_seed_lidar_data_path, 6))
        seg_seed_lidar_data_info = self.get_lidar_info(seg_seed_lidar_data_path)
        tar_camera = self.tar_seq.camera[tar_camera_name]
        tar_lidar = self.tar_seq.lidar
        spherical_lidar_data = self.lidar2ego2spherical(seg_seed_lidar_data[:, :3], self.ori_seq.lidar.poses[seg_seed_lidar_data_info['seq_idx']])
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

    def get_project_center(self, project_point2d, ori_lidar_data_path):
        project_point2d = project_point2d.astype(np.uint32)
        left, right = min(project_point2d[:,0]) - 2, max(project_point2d[:,0]) + 2
        low, high = min(project_point2d[:,1]) - 25, max(project_point2d[:,1]) + 2
        mid_x, mid_y = (left+right) // 2, (low+high) // 2
        # str_wh = ori_lidar_data_path.split('/')[-1].split('_')[-2:]
        # ori_mid_x, ori_mid_y = int(str_wh[0]), int(str_wh[1])
        seed_mask_jpg = self.read_pic(ori_lidar_data_path, color_t=0)
        seed_mask = np.squeeze(np.dstack(np.where(seed_mask_jpg != 0)))
        seed_mask_jpg_mid_x = seed_mask_jpg.shape[0] // 2
        seed_mask_jpg_mid_y = seed_mask_jpg.shape[1] // 2
        mid_offset = np.array((mid_x-seed_mask_jpg_mid_x, mid_y-seed_mask_jpg_mid_y))
        seed_mask_texture = seed_mask + mid_offset

    def filter_overlap_pts(self, seg_seed_lidar_data, seed_mask_texture, seg_seed_lidar_data_info, tar_camera_name, tar_seq_idx):
        tar_camera = self.tar_seq.camera[tar_camera_name]
        tar_lidar = self.tar_seq.lidar
        points3d_lidar_xyz = tar_lidar.data[tar_seq_idx].to_numpy()[:, :3]
        tar_semseg = self.tar_seq.semseg[tar_seq_idx]
        projected_points2d, _, inner_indices = geometry.projection(lidar_points=points3d_lidar_xyz, 
                                                                camera_data=tar_camera[tar_seq_idx],
                                                                camera_pose=tar_camera.poses[tar_seq_idx],
                                                                camera_intrinsics=tar_camera.intrinsics,
                                                                filter_outliers=True)
        tar_camera_lidar_data = points3d_lidar_xyz[inner_indices]
        tar_camera_semseg = tar_semseg[inner_indices]
        tar_camera_lidar2spherical = self.lidar2ego2spherical(tar_camera_lidar_data, tar_lidar.poses[tar_seq_idx])
        seg_seed_lidar2spherical_data = self.lidar2ego2spherical(seg_seed_lidar_data, self.seg_seq.lidar.poses[seg_seed_lidar_data_info['seq_idx']])
        seg_cover_idx, tar_cover_idx, seg_mask_cover_pts = self.get_overlap_pts(projected_points2d,
                                                                                seed_mask_texture,
                                                                                tar_camera_lidar2spherical,
                                                                                seg_seed_lidar2spherical_data)
        seg_seed_lidar_data = seg_seed_lidar_data[seg_cover_idx]
        tar_camera_lidar_data = tar_camera_lidar_data[tar_cover_idx]
        points3d_lidar_xyz = np.delete(points3d_lidar_xyz, inner_indices, 0)
        points3d_lidar_xyz = np.concatenate((points3d_lidar_xyz, tar_camera_lidar_data, seg_seed_lidar_data), axis=0)
    
    def seed_texture(self, ori_lidar_data_path, seg_mask_cover_pts, mid_offset):
        seed_jpg = self.read_pic(ori_lidar_data_path, color_t=1)
        seg_mask_cover_pts = seg_mask_cover_pts - mid_offset
        for pts in seg_mask_cover_pts:
            seed_jpg[pts[0], pts[1]] = (0,0,0)

    def copy_paste(self, img_src, img_main, mask_src, center, brightness):
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
        img_zero[(center[1]-int(img_src.shape[0]/2)):(center[1]+int(img_src.shape[0]/2)),
                (center[0]-int(img_src.shape[1]/2)):(center[0]+int(img_src.shape[1]/2))] = 255
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
        img_ill = cv2.illuminationChange(img_main, img_zero, alpha=alpha, beta=alpha)
        return img_ill
    
    def read_pic(self, ori_lidar_data_path, color_t=1):
        if os.path.exists(ori_lidar_data_path + '.jpg'):
            seed_jpg = cv2.imread(ori_lidar_data_path + '.jpg', color_t)
        elif os.path.exists(ori_lidar_data_path + '.png'):
            seed_jpg = cv2.imread(ori_lidar_data_path + '.png', color_t)
        else:
            raise 'seed jpg path {} not found'.format(ori_lidar_data_path)
        return seed_jpg
    
    def get_overlap_pts(self, projected_points2d, seed_mask_texture, tar_camera_lidar2spherical, seg_seed_lidar2spherical_data):
        projected_points2d_l = projected_points2d.astype(np.uint32).tolist()
        seed_mask_texture_l = seed_mask_texture.tolist()
        tar_cover_idx = np.ones(tar_camera_lidar2spherical.shape[0], dtype=bool)
        seg_cover_idx = np.ones(seg_seed_lidar2spherical_data.shape[0], dtype=bool)
        seg_mask_cover_idx = np.zeros(seed_mask_texture.shape[0], dtype=bool)
        for i in range(len(projected_points2d_l)):
            pts = projected_points2d_l[i]
            if pts in seed_mask_texture_l:
                pts_idx = seed_mask_texture_l.index(pts)
                tar_camera_lidar2spherical_idx = tar_camera_lidar2spherical[i]
                seg_lidar2spherical_data_idx = seg_seed_lidar2spherical_data[pts_idx]
                if tar_camera_lidar2spherical_idx[2] >= seg_lidar2spherical_data_idx[2]:
                    tar_cover_idx[i] = False
                else:
                    seg_cover_idx[pts_idx] = False
                    seg_mask_cover_idx[pts_idx] = True
        seg_mask_cover_pts = self.filter_mask_overlap(seg_mask_cover_idx, seed_mask_texture)
        return seg_cover_idx, tar_cover_idx, seg_mask_cover_pts

    
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
        lidar2ego_data = geometry.lidar_points_to_ego(ori_lidar_data[:, :3], self.ori_seq.lidar.poses[ori_lidar_data_info['seq_idx']])
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
    seg_aug.spherical_coord_ego_lidar_paste(lidar_data_path, 'back_camera', 0)


if __name__ == '__main__':
    test()