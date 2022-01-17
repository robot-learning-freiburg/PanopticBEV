import torch
import numpy as np


def compute_M(scale, image_size, bev_focal_length, bev_camera_z):
    """ image_size --> (H, W) """
    # Compute the mapping matrix from road to world (2D -> 3D)
    px_per_metre = abs((bev_focal_length * scale) / (bev_camera_z))

    # shift --> (W, H) (Where you want the output to be placed at wrt the input dimension)
    shift = ((image_size[1] / 2 * scale), image_size[0] * scale)  # Shift so that the thing is in Bottom center

    M = np.array([[1 / px_per_metre, 0, -shift[0] / px_per_metre],
                  [0, 1 / px_per_metre, -shift[1] / px_per_metre],
                  [0, 0, 0],  # This must be all zeros to cancel out the effect of Z
                  [0, 0, 1]])

    return M


def compute_intrinsic_matrix(fx, fy, px, py, img_scale):
    K = np.array([[fx * img_scale, 0, px * img_scale],
                  [0, fy * img_scale, py * img_scale],
                  [0, 0, 1]])
    return K


def compute_extrinsic_matrix(translation, rotation):
    # World to camera
    theta_w2c_x = np.deg2rad(rotation[0])
    theta_w2c_y = np.deg2rad(rotation[1])
    theta_w2c_z = np.deg2rad(rotation[2])

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_w2c_x), -np.sin(theta_w2c_x)],
                    [0, np.sin(theta_w2c_x), np.cos(theta_w2c_x)]], dtype=np.float)
    R_y = np.array([[np.cos(theta_w2c_y), 0, np.sin(theta_w2c_y)],
                    [0, 1, 0],
                    [-np.sin(theta_w2c_y), 0, np.cos(theta_w2c_y)]], dtype=np.float)
    R_z = np.array([[np.cos(theta_w2c_z), -np.sin(theta_w2c_z), 0],
                    [np.sin(theta_w2c_z), np.cos(theta_w2c_z), 0],
                    [0, 0, 1]], dtype=np.float)

    R = (R_y @ (R_x @ R_z))

    t = -np.array(translation, dtype=np.float)
    t_rot = np.matmul(R, np.expand_dims(t, axis=1))

    extrinsic = np.zeros((3, 4), dtype=np.float)
    extrinsic[:3, :3] = R[:3, :3]
    extrinsic[:, 3] = t_rot.squeeze(1)

    return extrinsic


def compute_homography(intrinsic_matrix, extrinsic_matrix, M):
    P = np.matmul(intrinsic_matrix, extrinsic_matrix)
    H = np.linalg.inv(P.dot(M))

    return H


def get_init_homography(intrinsics, extrinsics, bev_params, img_scale, img_size):
    extrinsic_mat = compute_extrinsic_matrix(extrinsics['translation'], extrinsics['rotation'])
    intrinsic_mat = compute_intrinsic_matrix(intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2],
                                             img_scale)
    M = compute_M(img_scale, img_size, bev_params['f'], bev_params['cam_z'])
    H = compute_homography(intrinsic_mat, extrinsic_mat, M)
    H = torch.tensor(H.astype(np.float32))
    return H