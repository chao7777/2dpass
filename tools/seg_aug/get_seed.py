from multiprocessing import Pool
from loguru import logger
import traceback
from tqdm import tqdm
import numpy as np
import os
from sklearn.cluster import DBSCAN
import cv2
import open3d
from tools.pandaset import DataSet
from tools.pandaset import geometry


def write_data(datas, save_dir, save_name, split_num=6):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f = open(os.path.join(save_dir, save_name), 'w')
    for data in datas:
        write_line = ''
        save_len = min(len(data), split_num)
        for i in range(save_len):
            write_line += (str(data[i])+'\t')
        f.write(write_line+'\n')
        f.flush()
    f.close()

def read_data(data_path, split_num=6):
    f = open(data_path)
    read_lines = f.readlines()
    f.close()
    data = []
    for line in read_lines:
        line_split = line.strip('\n').split('\t')
        assert len(line_split) >= split_num, 'The length read exceeds'
        split_l = []
        for i in range(split_num):
            split_l.append(float(line_split[i]))
        data.append(split_l)
    return np.array(data)

class MultiProcess:
    def __init__(self):
        pass
    
    def _finish(self):
        logger.info('finish _core_comp')
    
    def _print_error(value):
        print("线程池出错,出错原因为: ", value)

    def _core_comp(self, sub_task_num, sub_task, task_func, kwargs=None):
        try:
            logger.info("start to run sub_task_num {}, len(sub_task)  is {}",
                        sub_task_num, len(sub_task))
            if kwargs == None:
                task_func(sub_task, sub_task_num)
            else:
                task_func(sub_task, sub_task_num, kwargs)
        except:
            logger.error(traceback.format_exc())
            logger.error("comp error!")

    @staticmethod
    def multi_process_entr(self, tasks_enr, num, task_func, kwargs=None):
        try:
            err_res_list = []
            pool = Pool(num)
            tasks = []
            for i in range(num):
                tasks.append([])
            for i in range(len(tasks_enr)):
                tasks[i % num].append(tasks_enr[i])
            for i in range(0, len(tasks)):
                sub_task = tasks[i]
                sub_task_num = i
                res = pool.apply_async(self._core_comp,
                                    args=(MultiProcess, sub_task_num, sub_task, task_func, kwargs),
                                    callback=self._finish,
                                    error_callback=self._print_error)
        except:
            if len(err_res_list) > 0:
                raise err_res_list[0]
            else:
                raise
        finally:
            pool.close()
            pool.join()

def task_func(task, process_num, kwargs):
        print(task, process_num, kwargs)

def test_multi_process():
    tasks = [1,2]
    kwargs = {'other_kwargs':[]}
    MultiProcess.multi_process_entr(MultiProcess, tasks, 2, 'test', task_func, kwargs)

class GetSeed:
    def __init__(self, dataset_path, process_num, save_seed_jpg_dir, label_seed):
        self.dataset_path = dataset_path
        self.process_num = process_num
        self.save_seed_jpg_dir = save_seed_jpg_dir
        self.cameras = ['back_camera', 'front_camera', 'front_left_camera', 'front_right_camera', 'left_camera', 'right_camera']
        # self.cameras = ['front_left_camera']
        self.label_seed = label_seed
    
    def load_dataset(self):
        self.seq_list = []
        dataset = DataSet(self.dataset_path)
        sub_dirs = os.listdir(self.dataset_path)
        for sub in sub_dirs:
            if '053' not in sub:continue
            seq = dataset[sub]
            seq.load()
            self.seq_list.append({
                'seq':seq,
                'path':os.path.join(self.dataset_path, sub)
            })
    
    def get_seq_num(self, sub_dir):
        dir = os.path.join(sub_dir, 'lidar')
        paths = os.listdir(dir)
        num = 0
        for path in paths:
            if 'pkl.gz' not in path:continue
            num += 1
        return num

    def get_seed(self, seq_num):
        seq_num = min(seq_num, len(self.seq_list))
        # self.task_func(self.seq_list[:seq_num], 0, '')
        MultiProcess.multi_process_entr(MultiProcess, 
                                        self.seq_list[:seq_num], 
                                        self.process_num, 
                                        self.save_seed_jpg_dir,
                                        self.task_func,
                                        )
    
    def task_func(self, seq_list, num_process, save_seed_jpg_dir):
        for seq in seq_list:
            seq_num = self.get_seq_num(seq['path'])
            seq_sub_dir = seq['path'].split('/')[-1]
            for idx in tqdm(range(seq_num)):
                self.lidar_project_2d(seq['seq'], idx, seq_sub_dir)

    def get_cluster_point(self, label_data, cluster_label, camera_data, camera_pose, camera_intrinsics, camera_name, seq_sub_dir, label, seq_idx):
        save_jpg_dir = os.path.join(self.save_seed_jpg_dir, seq_sub_dir, '%02d'%(seq_idx), camera_name)
        if not os.path.exists(save_jpg_dir):
            os.makedirs(save_jpg_dir)
        img = np.array(camera_data)[...,::-1]
        for i in range(max(cluster_label)+1):
            cluster_labels_flag = np.ones(label_data.shape[0], dtype=bool)
            cluster_labels_flag = np.logical_and(cluster_labels_flag, np.array(cluster_label)==i)
            cluster_label_data = label_data[cluster_labels_flag]
            if np.min(cluster_label_data[:,2]) > 0.2:continue
            projected_points2d, _, _ = geometry.projection(lidar_points=cluster_label_data[:, :3], 
                                                            camera_data=camera_data,
                                                            camera_pose=camera_pose,
                                                            camera_intrinsics=camera_intrinsics,
                                                            filter_outliers=True)
            projected_points2d = projected_points2d.astype(np.uint32)
            left, right = min(projected_points2d[:,0]) - 2, max(projected_points2d[:,0]) + 2
            low, high = min(projected_points2d[:,1]) - 25, max(projected_points2d[:,1]) + 2
            left, right = max(0, left), min(img.shape[1], right)
            low, high = max(0, low), min(img.shape[0], high)
            mid_x, mid_y = (left+right) // 2, (low+high) // 2
            cluster_label_img = img[low:high, left:right]
            cv2.imwrite(os.path.join(save_jpg_dir, 'label_{}_{}_{}_{}.jpg'.format(label, i, mid_x, mid_y)), cluster_label_img)
            write_data(cluster_label_data, save_jpg_dir, 'label_{}_{}_{}_{}'.format(label, i, mid_x, mid_y))

        
    def lidar_project_2d(self, seq, seq_idx, seq_sub_dir):
        lidar = seq.lidar
        # lidar_pcd = open3d.geometry.PointCloud()
        points3d_lidar_xyz = lidar.data[seq_idx].to_numpy()
        semseg = seq.semseg[seq_idx].to_numpy()
        camera = seq.camera
        assert points3d_lidar_xyz.shape[0] == semseg.shape[0]
        for camera_name in self.cameras:
            _, _, inner_indices = geometry.projection(lidar_points=points3d_lidar_xyz[:, :3], 
                                                        camera_data=camera[camera_name][seq_idx],
                                                        camera_pose=camera[camera_name].poses[seq_idx],
                                                        camera_intrinsics=camera[camera_name].intrinsics,
                                                        filter_outliers=True)
            camera_semseg = semseg[inner_indices].flatten()
            xyz = points3d_lidar_xyz[inner_indices]
            for label in self.label_seed:
                label_bool = np.ones(camera_semseg.shape[0], dtype=bool)
                label_bool = np.logical_and(label_bool, camera_semseg==label)
                label_data = xyz[label_bool]
                if label_data.shape[0] < 15:continue
                clustering = DBSCAN(eps=1.5, min_samples=15).fit(label_data[:, :3])
                # lidar_pcd.points = open3d.utility.Vector3dVector(label_data)
                # lidar_pcd.translate((-5, 0, 0), relative=True)
                cluster_label = clustering.labels_
                self.get_cluster_point(label_data,
                                       cluster_label,
                                       camera[camera_name][seq_idx],
                                       camera[camera_name].poses[seq_idx],
                                       camera[camera_name].intrinsics,
                                       camera_name,
                                       seq_sub_dir,
                                       label,
                                       seq_idx)

def get_seed():
    dataset_path = '/share/qi.chao/open_sor_data/pandaset/pandaset_1'
    get_seed = GetSeed('/share/qi.chao/open_sor_data/pandaset/pandaset_1',
                        1,
                        '/share/qi.chao/open_sor_data/pandaset/seed_jpg',
                        [5])
    get_seed.load_dataset()
    get_seed.get_seed(1)


if __name__ == '__main__':
    get_seed()
