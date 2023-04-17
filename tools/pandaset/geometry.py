import numpy as np
import transforms3d as t3d

from .sensors import Lidar
from .sensors import Camera


def _heading_position_to_mat(heading, position):
    quat = np.array([heading["w"], heading["x"], heading["y"], heading["z"]])
    pos = np.array([position["x"], position["y"], position["z"]])
    transform_matrix = t3d.affines.compose(np.array(pos),
                                           t3d.quaternions.quat2mat(quat),
                                           [1.0, 1.0, 1.0])
    return transform_matrix


def projection(lidar_points, camera_data, camera_pose, camera_intrinsics, filter_outliers=True):
    camera_heading = camera_pose['heading']
    camera_position = camera_pose['position']
    camera_pose_mat = _heading_position_to_mat(camera_heading, camera_position)

    trans_lidar_to_camera = np.linalg.inv(camera_pose_mat)
    points3d_lidar = lidar_points
    points3d_camera = trans_lidar_to_camera[:3, :3] @ (points3d_lidar.T) + \
                        trans_lidar_to_camera[:3, 3].reshape(3, 1)

    K = np.eye(3, dtype=np.float64)
    K[0, 0] = camera_intrinsics.fx
    K[1, 1] = camera_intrinsics.fy
    K[0, 2] = camera_intrinsics.cx
    K[1, 2] = camera_intrinsics.cy

    inliner_indices_arr = np.arange(points3d_camera.shape[1])
    if filter_outliers:
        condition = points3d_camera[2, :] > 0.0
        points3d_camera = points3d_camera[:, condition]
        inliner_indices_arr = inliner_indices_arr[condition]

    points2d_camera = K @ points3d_camera
    points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T

    if filter_outliers:
        image_w, image_h = camera_data.size
        condition = np.logical_and(
            (points2d_camera[:, 1] < image_h) & (points2d_camera[:, 1] > 0),
            (points2d_camera[:, 0] < image_w) & (points2d_camera[:, 0] > 0))
        points2d_camera = points2d_camera[condition]
        points3d_camera = (points3d_camera.T)[condition]
        inliner_indices_arr = inliner_indices_arr[condition]
    return points2d_camera, points3d_camera, inliner_indices_arr


def lidar_points_to_ego(points, lidar_pose):
    lidar_pose_mat = _heading_position_to_mat(
        lidar_pose['heading'], lidar_pose['position'])
    transform_matrix = np.linalg.inv(lidar_pose_mat)
    return (transform_matrix[:3, :3] @ points.T +  transform_matrix[:3, [3]]).T

def ego_to_lidar_point(points, lidar_pose):
    lidar_pose_mat = _heading_position_to_mat(
        lidar_pose['heading'], lidar_pose['position'])
    transform_matrix = np.linalg.inv(lidar_pose_mat)
    return (np.linalg.inv(transform_matrix[:3, :3]) @ (points.T - transform_matrix[:3, [3]])).T


def center_box_to_corners(box):
    pos_x, pos_y, pos_z, dim_x, dim_y, dim_z, yaw = box
    half_dim_x, half_dim_y, half_dim_z = dim_x/2.0, dim_y/2.0, dim_z/2.0
    corners = np.array([[half_dim_x, half_dim_y, -half_dim_z],
                        [half_dim_x, -half_dim_y, -half_dim_z],
                        [-half_dim_x, -half_dim_y, -half_dim_z],
                        [-half_dim_x, half_dim_y, -half_dim_z],
                        [half_dim_x, half_dim_y, half_dim_z],
                        [half_dim_x, -half_dim_y, half_dim_z],
                        [-half_dim_x, -half_dim_y, half_dim_z],
                        [-half_dim_x, half_dim_y, half_dim_z]])
    transform_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0, pos_x],
        [np.sin(yaw), np.cos(yaw), 0, pos_y],
        [0, 0, 1.0, pos_z],
        [0, 0, 0, 1.0],
    ])
    corners = (transform_matrix[:3, :3] @ corners.T + transform_matrix[:3, [3]]).T
    return corners

def world2spherical(world_lidar_data):
    r = np.sqrt(np.sum(np.square(world_lidar_data), axis=1))
    thera = np.rad2deg(np.arccos(world_lidar_data[:,2]/r))
    phi = np.rad2deg(np.arctan2(world_lidar_data[:,1], world_lidar_data[:,0]))
    return np.squeeze(np.dstack((phi, thera, r)))

def spherical2world(spherical_lidar_data):
    spherical_lidar_data[:,0] = np.deg2rad(spherical_lidar_data[:,0])
    spherical_lidar_data[:,1] = np.deg2rad(spherical_lidar_data[:,1])
    z = spherical_lidar_data[:,2] * np.cos(spherical_lidar_data[:,1])
    y = spherical_lidar_data[:,2] * np.sin(spherical_lidar_data[:,1]) * np.sin(spherical_lidar_data[:,0])
    x = spherical_lidar_data[:,2] * np.sin(spherical_lidar_data[:,1]) * np.cos(spherical_lidar_data[:,0])
    return np.squeeze(np.dstack((x,y,z)))


if __name__ == '__main__':
    pass
