import numpy as np
import cv2
import json
import os
from tools.pandaset import DataSet
from tools.pandaset import geometry
import pickle
import gzip
from PIL import Image
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon,Point
import alphashape
import open3d
from tqdm import tqdm
from tools.pandaset.sensors import Intrinsics
import random

def dataset_test():
    from dataloader.dataset import get_model_class, get_collate_class
    from dataloader.pc_dataset import get_pc_model_class

    from main import parse_config
    config = parse_config()
    pc_dataset = get_pc_model_class(config['dataset_params']['pc_dataset_type'])
    dataset_type = get_model_class(config['dataset_params']['dataset_type'])
    train_config = config['dataset_params']['train_data_loader']
    val_config = config['dataset_params']['val_data_loader']
    train_pt_dataset = pc_dataset(config, data_path=train_config['data_path'], imageset='train')
    dataset=dataset_type(train_pt_dataset, config, train_config, max_dropout_ratio=0).__getitem__(0)

def dataset_test_c():
    from mmcv import Config
    from mmcv.utils import build_from_cfg
    from dataloader.build import PIPELINE
    cfg = Config.fromfile('config/fusion/test.py')
    pipeline = build_from_cfg(cfg.pipeline, PIPELINE)

def pandaset2pcd():
    pkl_file = '/share/qi.chao/open_sor_data/pandaset/pandaset_0/002/lidar/32.pkl.gz'
    semseg_file = '/share/qi.chao/open_sor_data/pandaset/pandaset_0/002/annotations/semseg/32.pkl.gz'
    data = pickle.load(gzip.open(pkl_file, "rb")).values
    semseg_data = pickle.load(gzip.open(semseg_file, "rb")).values
    veg = np.ones(semseg_data.shape[0], dtype=bool)
    veg = np.logical_and(veg, semseg_data[:,0]==5)
    data = data[veg]
    point_num = data.shape[0]
    pcd_file = 'work_dirs/pandaset/pcd/002_32_veg_pcd.pcd'
    handle = open(pcd_file, 'a')
    handle.write(
    '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
    string = '\nWIDTH ' + str(point_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    handle.write(string)
    handle.write('\nDATA ascii')
    for i in range(point_num):
        string = '\n' + str(data[i, 0]) + ' ' + str(data[i, 1]) + ' ' + str(data[i, 2])
        handle.write(string)
    handle.close()

def load():
    dataset = DataSet("/share/qi.chao/open_sor_data/pandaset/pandaset_1")
    seq053 = dataset["053"]
    seq053.load()

def vis(seq_idx, camera_name, seq, label):
    lidar = seq.lidar
    points3d_lidar_xyz = lidar.data[seq_idx].to_numpy()[:, :3]
    choosen_camera = seq.camera[camera_name]
    projected_points2d, camera_points_3d, inner_indices = geometry.projection(lidar_points=points3d_lidar_xyz, 
                                                                          camera_data=choosen_camera[seq_idx],
                                                                          camera_pose=choosen_camera.poses[seq_idx],
                                                                          camera_intrinsics=choosen_camera.intrinsics,
                                                                          filter_outliers=True)
    img = np.array(seq.camera[camera_name][seq_idx])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('work_dirs/pandaset/jpg/ori_jpg/{}_{}.jpg'.format(seq_idx, camera_name), img)
    semseg = seq.semseg[seq_idx].to_numpy()
    semseg_on_image = semseg[inner_indices].flatten()
    veg = np.ones(semseg_on_image.shape[0], dtype=bool)
    veg = np.logical_and(veg, semseg_on_image==label)
    veg_pts = projected_points2d[veg]
    print(len(veg_pts))
    for pts in veg_pts:
        cv2.circle(img, (int(pts[0]), int(pts[1])), 1, (0,255,0), 4)
    cv2.imwrite('work_dirs/pandaset/jpg/veg/{}_{}.jpg'.format(seq_idx, camera_name), img)

def read(path):
    f = open(path)
    lines = f.readlines()
    f.close()
    return lines

def write(path, lines, write_option='w'):
    f = open(path, write_option)
    f.writelines(lines)
    f.close()


def vis_clust_label(lidar_data, xyz, cluster_labels, camera_data, camera_pose, camera_intrinsics, cluster_label, label_name):
    save_jpg_dir = 'work_dirs/pandaset/jpg/{}/clust_label'.format(label_name)
    if not os.path.exists(save_jpg_dir):
        os.makedirs(save_jpg_dir)
    assert xyz.shape[0] == len(cluster_labels)
    labels_flag = np.ones(xyz.shape[0], dtype=bool)
    labels_flag = np.logical_and(labels_flag, np.array(cluster_labels)==cluster_label)
    label_data = xyz[labels_flag]
    lidar_data = lidar_data[labels_flag]
    projected_points2d, _, _ = geometry.projection(lidar_points=label_data, 
                                                    camera_data=camera_data,
                                                    camera_pose=camera_pose,
                                                    camera_intrinsics=camera_intrinsics,
                                                    filter_outliers=True)
    img = np.array(camera_data)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    for pts in projected_points2d:
        cv2.circle(img, (int(pts[0]), int(pts[1])), 1, (0,255,0), 4)
    cv2.imwrite(os.path.join(save_jpg_dir, '{}.jpg'.format(str(cluster_label))), img)
    write_data(lidar_data, 'work_dirs/pandaset/jpg/{}/clust_label_data'.format(label_name), 'label_{}_lidar_data'.format(cluster_label))
    project_point2d_write(projected_points2d, cluster_label, 'work_dirs/pandaset/jpg/{}/project_point_data'.format(label_name))


def project_point2d_write(point, label, save_dir='work_dirs/pandaset/jpg/project_point_data'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f = open(os.path.join(save_dir, 'label_{}_project_point_data'.format(label)), 'w')
    for pts in point:
        writw_line = str(pts[0])+'\t'+str(pts[1])+'\n'
        f.write(writw_line)
        f.flush()
    f.close()


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


def test_one():
    label, label_name = 5, 'vegetation'
    lidar_data = pickle.load(gzip.open('/share/qi.chao/open_sor_data/pandaset/pandaset_1/053/lidar/00.pkl.gz'))
    semseg_data = pickle.load(gzip.open('/share/qi.chao/open_sor_data/pandaset/pandaset_1/053/annotations/semseg/00.pkl.gz')).values
    xyz = lidar_data.values[:, :3]
    camera_data = Image.open('/share/qi.chao/open_sor_data/pandaset/pandaset_1/053/camera/back_camera/00.jpg')
    camera_pose = json.load(open('/share/qi.chao/open_sor_data/pandaset/pandaset_1/053/camera/back_camera/poses.json'))[0]
    camera_intrinsics = json.load(open('/share/qi.chao/open_sor_data/pandaset/pandaset_1/053/camera/back_camera/intrinsics.json'))
    camera_intrinsics = Intrinsics(fx=camera_intrinsics['fx'],
                                   fy=camera_intrinsics['fy'],
                                   cx=camera_intrinsics['cx'],
                                   cy=camera_intrinsics['cy'])
    projected_points2d, camera_points_3d, inner_indices = geometry.projection(lidar_points=xyz, 
                                                                          camera_data=camera_data,
                                                                          camera_pose=camera_pose,
                                                                          camera_intrinsics=camera_intrinsics,
                                                                          filter_outliers=True)
    semseg_data = semseg_data[inner_indices].flatten()
    xyz = xyz[inner_indices]
    lidar_data = lidar_data.values[inner_indices]
    write_data(lidar_data, 'work_dirs/pandaset/jpg/data', 'back_camera_lidar', 6)
    veg = np.ones(semseg_data.shape[0], dtype=bool)
    veg = np.logical_and(veg, semseg_data==label)
    veg_pts = projected_points2d[veg]
    veg_lidar = xyz[veg]
    lidar_data = lidar_data[veg]
    img = np.array(camera_data)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    for pts in veg_pts:
        cv2.circle(img, (int(pts[0]), int(pts[1])), 1, (0,255,0), 4)
    cv2.imwrite('work_dirs/pandaset/jpg/label/{}.jpg'.format(label_name), img)
    project_point2d_write(veg_pts, label_name, 'work_dirs/pandaset/jpg/label')
    write_data(lidar_data, 'work_dirs/pandaset/jpg/label', 'label_{}_lidar_data'.format(label_name))
    clustering = DBSCAN(eps=1.5, min_samples=15).fit(veg_lidar)
    cluster_labels = clustering.labels_
    for i in range(max(cluster_labels)+1):
        vis_clust_label(lidar_data, veg_lidar, cluster_labels, camera_data, camera_pose, camera_intrinsics, i, label_name)

def get_semseg_num():
    a = {'y':[], 'n':[]}
    dir = '/share/qi.chao/open_sor_data/pandaset'
    sub_dirs = ['pandaset_0']
    for sub_dir in sub_dirs:
        full_dir = os.path.join(dir, sub_dir)
        paths = os.listdir(full_dir)
        for path in tqdm(paths):
            semseg_path = os.path.join(full_dir, path, 'annotations')
            if 'annotations' not in os.listdir(os.path.join(full_dir, path)):continue
            if 'semseg' in os.listdir(semseg_path):
                a['y'].append(path)
            else:
                a['n'].append(path)
    for key in a:
        print(list(set(a[key])))
        print(len(list(set(a[key]))))
    random_sample = random.sample(a['y'], 7)
    print(random_sample)

def grabcut():
    img = cv2.imread('work_dirs/pandaset/jpg/ori_jpg/0_back_camera.jpg')
    res_lines = read('work_dirs/pandaset/jpg/car/project_point_data/label_4_project_point_data')
    project_point_data = []
    mask = np.zeros(img.shape[:2], np.uint8)
    for line in res_lines:
        line_split = line.strip('\n').split('\t')
        project_point_data.append([float(line_split[0]), float(line_split[1])])
        # mask[int(float(line_split[1])), int(float(line_split[0]))] = 1
    alpha_shape=alphashape.alphashape(project_point_data,0)
    alpha_shape = list(alpha_shape.exterior.coords)
    polygon = Polygon(alpha_shape)
    for x_idx in range(mask.shape[1]):
        for y_idx in range(mask.shape[0]):
            if polygon.contains(Point(x_idx, y_idx)):
                mask[y_idx, x_idx] = 1
    project_point_data = np.array(project_point_data).astype(np.uint32)
    left, right = min(project_point_data[:,0]) - 2, max(project_point_data[:,0]) + 2
    low, high = min(project_point_data[:,1]) - 2, max(project_point_data[:,1]) + 2
    # img = img[low:high, left:right]
    # mask = mask[low:high, left:right]
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (left,low, right-left, high-low)
    # cv2.imwrite('work_dirs/pandaset/jpg/car/grabcut/crop.jpg', img)
    cv2.imwrite('work_dirs/pandaset/jpg/car/grabcut/mask.jpg', mask)
    # (mask, bgModel, fgModel) = cv2.grabCut(img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
    (mask, bgModel, fgModel) = cv2.grabCut(img,mask,None,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask==0)|(mask==2),0,1).astype('uint8')
    img_g = img*mask2[:,:,np.newaxis]
    cv2.imwrite('work_dirs/pandaset/jpg/car/grabcut/grab.jpg', img_g)

def grabcut_mask():
    img = cv2.imread('work_dirs/pandaset/jpg/ori_jpg/0_back_camera.jpg')
    new_maks = cv2.imread('work_dirs/pandaset/jpg/grabcut/sam_mask.jpg', 0)
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    mask[new_maks == 0] = 0
    mask[new_maks == 255] = 1
    mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask[:,:,np.newaxis]
    cv2.imwrite('work_dirs/pandaset/jpg/grabcut/grab.jpg', img)

def bitwise_grab():
    # 加载图像与掩膜求交集抠图
    img = cv2.imread('work_dirs/pandaset/jpg/ori_jpg/0_back_camera.jpg')
    mask = cv2.imread('work_dirs/pandaset/jpg/grabcut/sam_mask.jpg')
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    res = cv2.bitwise_and(img, img, mask=mask_gray)
    cv2.imwrite('work_dirs/pandaset/jpg/grabcut/bitwise_grab.jpg', res)

def polygon():
    import alphashape
    from shapely.geometry import Polygon,Point
    img = cv2.imread('work_dirs/pandaset/jpg/ori_jpg/0_back_camera.jpg')
    res_lines = read('work_dirs/pandaset/jpg/car/project_point_data/label_5_project_point_data')
    project_point_data = []
    for line in res_lines:
        line_split = line.strip('\n').split('\t')
        project_point_data.append([float(line_split[0]), float(line_split[1])])
    alpha_shape=alphashape.alphashape(project_point_data,0)
    alpha_shape = list(alpha_shape.exterior.coords)
    polygon = Polygon(alpha_shape)
    x, y = 290, 640
    point = Point(x,y)
    print(polygon.contains(point))
    for pts in alpha_shape:
        cv2.circle(img, (int(pts[0]), int(pts[1])), 1, (0,255,0),4)
    cv2.circle(img, (x,y), 1, (0,0,255),4)
    cv2.imwrite('work_dirs/pandaset/jpg/car/grabcut/polygon.jpg', img)


def translate_lidar_data(x,y,z, a,b,c):
    img = cv2.imread('work_dirs/pandaset/jpg/ori_jpg/0_back_camera.jpg')
    camera_data = Image.open('/share/qi.chao/open_sor_data/pandaset/pandaset_1/053/camera/back_camera/00.jpg')
    camera_pose = json.load(open('/share/qi.chao/open_sor_data/pandaset/pandaset_1/053/camera/back_camera/poses.json'))[0]
    camera_intrinsics = json.load(open('/share/qi.chao/open_sor_data/pandaset/pandaset_1/053/camera/back_camera/intrinsics.json'))
    lidar_pose = json.load(open('/share/qi.chao/open_sor_data/pandaset/pandaset_1/053/lidar/poses.json'))[0]
    camera_intrinsics = Intrinsics(fx=camera_intrinsics['fx'],
                                   fy=camera_intrinsics['fy'],
                                   cx=camera_intrinsics['cx'],
                                   cy=camera_intrinsics['cy'])
    lidar_pcd = open3d.geometry.PointCloud()
    lidar_data = []
    read_lines = read('work_dirs/pandaset/jpg/vegetation/clust_label_data/label_3_lidar_data')
    for line in read_lines:
        line_split = line.strip('\n').split('\t')
        lidar_data.append([float(line_split[0]), float(line_split[1]), float(line_split[2])])
    lidar_data = np.array(lidar_data)
    lidar_pcd.points = open3d.utility.Vector3dVector(lidar_data)
    # lidar_pcd.translate((x, y, z), relative=True)
    lidar_center = lidar_pcd.get_center()
    lidar_tran_center = lidar_center + np.array([x,y,z])
    lidar_pcd.translate(lidar_tran_center, relative=False)
    R = lidar_pcd.get_rotation_matrix_from_xyz(rotation = [a, b, np.radians(c)])
    lidar_pcd.rotate(R=R, center=lidar_tran_center)
    translate_data = np.array(lidar_pcd.points)
    ego2lidar = geometry.ego_to_lidar_point(translate_data, lidar_pose)
    write_data(ego2lidar, 'work_dirs/pandaset/jpg/translate/world/pcd', '{}_{}_{}_{}_{}_{}'.format(x,y,z,a,b,c), 3)
    projected_points2d, camera_points_3d, inner_indices = geometry.projection(lidar_points=translate_data, 
                                                                          camera_data=camera_data,
                                                                          camera_pose=camera_pose,
                                                                          camera_intrinsics=camera_intrinsics,
                                                                          filter_outliers=True)
    for pts in projected_points2d:
        cv2.circle(img, (int(pts[0]), int(pts[1])), 1, (0,255,0), 4)
    
    cv2.imwrite('work_dirs/pandaset/jpg/translate/rotate/world/translate_{}_{}_{}.jpg'.format(x,y,z), img)

def get_data(path, num=2):
    read_lines = read(path)
    data = []
    for line in read_lines:
        line_split = line.strip('\n').split('\t')
        if num == 2:
            data.append([float(line_split[0]), float(line_split[1])])
        elif num == 3:
            data.append([float(line_split[0]), float(line_split[1]), float(line_split[2])])
        else:
            raise 'number error'
    return np.array(data)
    
def get_sam_mask():
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    sam_checkpoint = "/share/qi.chao/open_sor_data/checkpoint/sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "default"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    img = cv2.imread('/home/qi.chao/c7/mycode/2dpass/work_dirs/pandaset/jpg/ori_jpg/0_back_camera.jpg')
    masks = mask_generator.generate(img)
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    mask = np.zeros((img.shape[0], img.shape[1], 3))
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random((1, 3))*255
        color_mask = color_mask.tolist()[0]
        if m[400,1650] == False:continue
        m_true = np.where(m == True)
        m_true = np.dstack(m_true)
        for pts in m_true[0]:
            pts_l = pts.tolist()
            mask[pts_l[0], pts_l[1]] = [255,255,255]
    cv2.imwrite('/home/qi.chao/c7/mycode/2dpass/work_dirs/pandaset/jpg/grabcut/sam_mask.jpg', mask)


def lidar2ego():
    num, label = 4, 'car'
    lidar_path = 'work_dirs/pandaset/jpg/{}/clust_label_data/label_{}_lidar_data'.format(label, num)
    lidar_data = get_data(lidar_path, 3)
    lidar_pose = json.load(open('/share/qi.chao/open_sor_data/pandaset/pandaset_1/053/lidar/poses.json'))[0]
    ego_pandar64_points = geometry.lidar_points_to_ego(lidar_data, lidar_pose)
    write_data(ego_pandar64_points, 'work_dirs/pandaset/jpg/{}/lidar2ego_data'.format(label), 'label_{}_lidar_data'.format(num), 3)


def ego2lidar():
    num, label = 7, 'building'
    lidar_path = 'work_dirs/pandaset/jpg/{}/lidar2ego_data/label_{}_lidar_data'.format(label, num)
    lidar_data = get_data(lidar_path, 3)
    lidar_pose = json.load(open('/share/qi.chao/open_sor_data/pandaset/pandaset_1/053/lidar/poses.json'))[0]
    lidar_data = geometry.ego_to_lidar_point(lidar_data, lidar_pose)
    write_data(lidar_data, 'work_dirs/pandaset/jpg/{}/lidar2ego_data'.format(label), 'label_{}_lidar_data'.format(num), 3)

def cal_spherical_dis():
    pt_spherical = np.array([-52.75542223,  85.36139588,  17.92341606])
    pt_world = geometry.spherical2world(pt_spherical)
    world_data = read_data('work_dirs/pandaset/jpg/vegetation/lidar2ego_data/label_3_lidar_data', 3)
    world2spherical_data = geometry.world2spherical(world_data)

def pts_project():
    spherical_data = np.array([[-121.60304053,83.57461298,17.8637065], [-122.69975364,77.27375102,18.52313402]])
    x,y,z=-1,-0.5,-0.3
    spherical_data = spherical_data + np.array([x,y,z])
    spherical2world_data = geometry.spherical2world(spherical_data)
    camera_data = Image.open('/share/qi.chao/open_sor_data/pandaset/pandaset_1/053/camera/back_camera/00.jpg')
    camera_pose = json.load(open('/share/qi.chao/open_sor_data/pandaset/pandaset_1/053/camera/back_camera/poses.json'))[0]
    camera_intrinsics = json.load(open('/share/qi.chao/open_sor_data/pandaset/pandaset_1/053/camera/back_camera/intrinsics.json'))
    camera_intrinsics = Intrinsics(fx=camera_intrinsics['fx'],
                                   fy=camera_intrinsics['fy'],
                                   cx=camera_intrinsics['cx'],
                                   cy=camera_intrinsics['cy'])
    lidar_pose = json.load(open(os.path.join('/share/qi.chao/open_sor_data/pandaset/pandaset_1/053', 'lidar', 'poses.json')))[0]
    ego2lidar_data = geometry.ego_to_lidar_point(spherical2world_data, lidar_pose)
    print(ego2lidar_data)
    projected_points2d, _, inner_indices = geometry.projection(lidar_points=ego2lidar_data, 
                                                            camera_data=camera_data,
                                                            camera_pose=camera_pose,
                                                            camera_intrinsics=camera_intrinsics,
                                                            filter_outliers=True)
    img = np.array(camera_data)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    for pts in projected_points2d:
        cv2.circle(img, (int(pts[0]), int(pts[1])), 1, (0,255,0), 4)
    cv2.imwrite('work_dirs/pandaset/jpg/translate/pts/spherical/{}_{}_{}.jpg'.format(x,y,z), img)

def lidar_vis():
    pcd = open3d.io.read_point_cloud('file_path')
    print(pcd)
    pcd.paint_uniform_color([0, 0, 1])#指定显示为蓝色
    #点云显示
    open3d.visualization.draw_geometries([pcd])

def repair(tar_paste_info, dataset_yaml):
    lidar_pcd = open3d.geometry.PointCloud()
    label = 26
    label_name = dataset_yaml['labels'][label]
    projec_2d, _, inner_indices = geometry.projection(lidar_points=tar_paste_info['lidar_data'][:, :3], 
                                                  camera_data=tar_paste_info['camera_data'],
                                                  camera_pose=tar_paste_info['calib_info']['camera_pose'],
                                                  camera_intrinsics=tar_paste_info['calib_info']['camera_intrinsics'],
                                                  filter_outliers=True)
    camera_seg_data = tar_paste_info['semseg'][inner_indices]
    camera_lidar_data = tar_paste_info['lidar_data'][inner_indices]
    
    paste_label = np.ones(camera_seg_data.shape[0], dtype=bool)
    paste_label = np.logical_and(paste_label, (camera_seg_data[:, 0] == label))
    label_lidar = camera_lidar_data[paste_label]
    clustering = DBSCAN(eps=1.5, min_samples=4).fit(label_lidar[:, :3])
    cluster_label = clustering.labels_
    cluster_labels_flag = np.ones(label_lidar.shape[0], dtype=bool)
    cluster_labels_flag = np.logical_and(cluster_labels_flag, np.array(cluster_label)==2)
    cluster_label_data = label_lidar[cluster_labels_flag]
    # cluster_label_data_ego = geometry.lidar_points_to_ego(cluster_label_data[:, :3], tar_paste_info['calib_info']['lidar_pose'])
    # lidar_pcd.points = open3d.utility.Vector3dVector(cluster_label_data_ego)
    # lidar_pcd.translate((-0.2, -0.2, 0), relative=True)
    # cluster_label_data_lidar = geometry.ego_to_lidar_point(np.array(lidar_pcd.points), tar_paste_info['calib_info']['lidar_pose'])
    # cluster_label_data_lidar = np.concatenate((cluster_label_data_lidar, cluster_label_data[:, 3:]), axis=1)
    # label_lidar[cluster_labels_flag] = cluster_label_data_lidar
    # camera_lidar_data[paste_label] = label_lidar
    # tar_paste_info['lidar_data'][inner_indices] = camera_lidar_data
    # df = pandas.DataFrame(tar_paste_info['lidar_data'])
    # df.rename(columns={0: "x", 1: "y", 2: "z", 3:'i', 4:'t', 5:'d'})
    # pickle.dump(df, gzip.open('work_dirs/pandaset/jpg/013_25_back_camera/repair_25.pkl.gz', 'wb'))
    projec_2d, _, inner_indices = geometry.projection(lidar_points=cluster_label_data[:, :3], 
                                                camera_data=tar_paste_info['camera_data'],
                                                camera_pose=tar_paste_info['calib_info']['camera_pose'],
                                                camera_intrinsics=tar_paste_info['calib_info']['camera_intrinsics'],
                                                filter_outliers=True)
    img = np.array(tar_paste_info['camera_data'])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for pts in projec_2d:
        cv2.circle(img, (int(pts[0]), int(pts[1])), 3, (0,255,0), 4)
    cv2.imwrite('work_dirs/pandaset/jpg/013_25_back_camera/bicycle_{}.jpg'.format(2), img)
    # img = np.array(tar_paste_info['camera_data'])
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)    
    # for pts in label_projec_2d:
    #     cv2.circle(img, (int(pts[0]), int(pts[1])), 3, (0,255,0), 4)
    # cv2.imwrite('work_dirs/pandaset/jpg/013_25_back_camera/{}.jpg'.format(label_name), img)

if __name__ == '__main__':
    dataset_test_c()
