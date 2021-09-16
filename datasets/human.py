"""
Author: Shengyu Huang
Last modified: 30.11.2020
"""
import json
import os, sys, glob, torch
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import open3d as o3d
from lib.benchmark_utils import to_o3d_pcd, to_tsfm, get_correspondences, to_tensor


class HumanHeadDataset(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """

    def __init__(self, sources, config, data_augmentation=True):
        super(HumanHeadDataset, self).__init__()
        self.sources = sources
        self.base_dir = config.root
        self.overlap_radius = config.overlap_radius
        self.data_augmentation = data_augmentation
        self.config = config

        self.rot_factor = 1.
        self.augment_noise = config.augment_noise
        self.max_points = 1500

    def __len__(self):
        # return len(self.sources)
        return 5

    def __getitem__(self, item):
        # get transformation
        rot = np.asarray(self.sources[item]['pose'])[:3, :3]
        trans = np.asarray(self.sources[item]['pose'])[:3, 3]
        tsfm = to_tsfm(rot, trans)

        # get pointcloud
        src_path = os.path.join(self.base_dir, self.sources[item]['pc_model'][1:])
        tgt_path = os.path.join(self.base_dir, self.sources[item]['pc_artificial'][1:])
        # src_pcd = torch.load(src_path)
        # tgt_pcd = torch.load(tgt_path)

        src_pcd = o3d.io.read_point_cloud(src_path)
        src_pcd = src_pcd.voxel_down_sample(self.config.voxel_down)
        src_pcd = np.asarray(src_pcd.points)
        # src_pcd /= 100.0

        tgt_pcd = o3d.io.read_point_cloud(tgt_path)
        tgt_pcd = tgt_pcd.voxel_down_sample(self.config.voxel_down)
        tgt_pcd = np.asarray(tgt_pcd.points)
        # tgt_pcd /= 100.0

        # Get matches
        # matching_inds = get_correspondences(tgt_pcd, tgt_pcd, tsfm, self.config.voxel_down*1.5)

        # if (matching_inds.size(0) < self.max_corr and self.split == 'train'):
        #     return self.__getitem__(np.random.choice(len(self.files), 1)[0])
        #
        # # if we get too many points, we do some downsampling
        # if (src_pcd.shape[0] > self.max_points):
        #     # print('     oversize: ', src_pcd.shape[0])
        #     idx = np.random.permutation(src_pcd.shape[0])[:self.max_points]
        #     src_pcd = src_pcd[idx, :]
        #     # print('     down to: ', src_pcd.shape[0])
        # if (tgt_pcd.shape[0] > self.max_points):
        #     # print('     oversize: ', tgt_pcd.shape[0])
        #     idx = np.random.permutation(tgt_pcd.shape[0])[:self.max_points]
        #     tgt_pcd = tgt_pcd[idx, :]
        #     # print('     down to: ', tgt_pcd.shape[0])

        # add gaussian noise
        self.data_augmentation = False
        if self.data_augmentation:
            # rotate the point cloud
            euler_ab = np.random.rand(3) * np.pi * 2 / self.rot_factor  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                src_pcd = np.matmul(rot_ab, src_pcd.T).T
                rot = np.matmul(rot, rot_ab.T)
            else:
                tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T
                rot = np.matmul(rot_ab, rot)
                trans = np.matmul(rot_ab, trans)

            src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * self.augment_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * self.augment_noise

        if (trans.ndim == 1):
            trans = trans[:, None]

        # get correspondence at fine level
        tsfm = to_tsfm(rot, trans)
        correspondences = get_correspondences(to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd), tsfm, self.overlap_radius)

        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        rot = rot.astype(np.float32)
        trans = trans.astype(np.float32)

        return src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, correspondences, src_pcd, tgt_pcd, torch.ones(1)
