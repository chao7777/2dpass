from multiprocessing import Pool
from loguru import logger
import traceback
import numpy as np
import os
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

    def _core_comp(self, sub_task_num, sub_task, save_dir, task_func, kwargs=None):
        try:
            logger.info("start to run sub_task_num {}, len(sub_task)  is {}",
                        sub_task_num, len(sub_task))
            if kwargs == None:
                task_func(sub_task, sub_task_num, save_dir)
            else:
                task_func(sub_task, sub_task_num, save_dir, kwargs)
        except:
            logger.error(traceback.format_exc())
            logger.error("comp error!")

    @staticmethod
    def multi_process_entr(self, tasks_enr, num, save_dir, task_func, kwargs=None):
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
                                    args=(MultiProcess, sub_task_num, sub_task, save_dir, task_func, kwargs),
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


def lidar2ego2spherical(lidar_data, lidar_pose):
    lidar2ego_data = geometry.lidar_points_to_ego(lidar_data, lidar_pose)
    spherical_lidar_data = geometry.world2spherical(lidar2ego_data)
    return spherical_lidar_data

def spherical2ego2lidar(spherical_lidar_data, lidar_pose):
    world_lidar_data = geometry.spherical2world(spherical_lidar_data)
    ego2lidar_data = geometry.ego_to_lidar_point(world_lidar_data, lidar_pose)
    return ego2lidar_data

def filter_lidar_data(lidar_data, semseg_data, labels, filter=True):
    assert lidar_data.shape[0] == semseg_data.shape[0]
    flag = np.zeros(lidar_data.shape[0], dtype=bool)
    for label in labels:
        flag = np.logical_or(flag, semseg_data[:, 0] == label)
    if filter:
        flag = ~flag
    return lidar_data[flag], semseg_data[flag]