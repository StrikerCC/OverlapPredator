"""
Author: Shengyu Huang
Last modified: 30.11.2020
"""
import copy
import json
import os, sys, glob, torch
import numpy as np
import open3d.cpu.pybind.visualization
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import open3d as o3d
from lib.benchmark_utils import to_o3d_pcd, to_tsfm, get_correspondences, to_tensor


class HumanDataset(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """

    def __init__(self, sources, config, data_augmentation=True):
        super(HumanDataset, self).__init__()
        self.sources = sources
        self.base_dir = config.root
        self.overlap_radius = config.overlap_radius
        self.data_augmentation = data_augmentation
        self.config = config

        self.rot_factor = 1.
        self.augment_noise = config.augment_noise
        self.max_points = 11600

    def __len__(self):
        # return len(self.sources)
        return len(self.sources)

    def __getitem__(self, item):
        # get transformation
        tf_src_2_tgt = np.asarray(self.sources[item]['tf_src_2_tgt']).astype(np.float32)
        rot = tf_src_2_tgt[:3, :3]
        trans = tf_src_2_tgt[:3, 3] / 1000.0
        tsfm = to_tsfm(rot, trans)

        # get pointcloud
        # src_path = os.path.join(self.base_dir, self.sources[item]['pc_src'][1:]) if self.sources[item]['pc_src'][0] == '.' else os.path.join(self.base_dir, self.sources[item]['pc_src'])
        # tgt_path = os.path.join(self.base_dir, self.sources[item]['pc_tgt'][1:]) if self.sources[item]['pc_tgt'][0] == '.' else os.path.join(self.base_dir, self.sources[item]['pc_tgt'])

        src_path = self.base_dir + self.sources[item]['pc_src'][1:] if self.sources[item]['pc_src'][0] == '.' else self.base_dir + self.sources[item]['pc_src']
        tgt_path = self.base_dir + self.sources[item]['pc_tgt'][1:] if self.sources[item]['pc_tgt'][0] == '.' else self.base_dir + self.sources[item]['pc_tgt']

        # src_pcd = torch.load(src_path)
        # tgt_pcd = torch.load(tgt_path)

        src_pcd = o3d.io.read_point_cloud(src_path)
        src_pcd = src_pcd.voxel_down_sample(self.config.voxel_down)

        tgt_pcd = o3d.io.read_point_cloud(tgt_path)
        tgt_pcd = tgt_pcd.voxel_down_sample(self.config.voxel_down)

        # '''vis to confirm'''
        # if item in {0, 1, 2}:
        #     src_pcd_temp = copy.deepcopy(src_pcd)
        #     src_pcd_temp.transform(tf_src_2_tgt)
        #     o3d.visualization.draw_geometries([src_pcd_temp, tgt_pcd])
        #     del src_pcd_temp

        src_pcd = np.asarray(src_pcd.points) / 1000.0
        tgt_pcd = np.asarray(tgt_pcd.points) / 1000.0
        np.random.shuffle(src_pcd)
        np.random.shuffle(tgt_pcd)

        if (trans.ndim == 1):
            trans = trans[:, None]

        # get correspondence at fine level
        correspondences = get_correspondences(to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd), tsfm, self.overlap_radius)

        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        rot = rot.astype(np.float32)
        trans = trans.astype(np.float32)

        return src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, correspondences, src_pcd, tgt_pcd, torch.ones(1)
