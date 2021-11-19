# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 11/19/21 4:16 PM
"""
import numpy as np
import copy
from cam import camera
import open3d as o3
import torch
from datasets.dataloader import get_dataloader
from datasets.human import HumanDataset
from configs.models import architectures
from models.architectures import KPFCNN
from lib.tester import get_trainer
from easydict import EasyDict as edict
from lib.utils import load_config, load_json
from lib.benchmark_utils import ransac_pose_estimation
from scripts.demo import ThreeDMatchDemo, draw_registration_result


def new_frame():
    """take a new frame"""
    frame_file_path = './pc_shut.ply'
    pc_file_path = './pc_range.ply'
    camera.hello()
    camera.shut()
    pc = o3.io.read_point_cloud(frame_file_path)
    pc = preprocess(pc)
    print(pc)
    print(pc.get_min_bound())
    print(pc.get_max_bound())
    o3.io.write_point_cloud(pc_file_path, pc)
    return pc_file_path


def preprocess(pc):
    """"""
    '''crop'''
    box = o3.geometry.AxisAlignedBoundingBox([-1000.0, -1000.0, 0.0], [1000.0, 1000.0, 3000.0])
    pc = o3.geometry.PointCloud.crop(pc, box)

    '''smoothing'''

    '''voxeldownsampling'''
    pc.scale(scale=0.001, center=[0, 0, 0])
    pc = o3.geometry.PointCloud.voxel_down_sample(pc, voxel_size=0.0002)
    return pc


def get_config():
    config = edict(load_config('./configs/train/head.yaml'))
    config.architecture = architectures['human']
    config.model = KPFCNN(config)
    checkpoint = torch.load('/home/cheng/proj/3d/3DPcReg/weights/model_best_recall.pth')
    config.model.load_state_dict(checkpoint['state_dict'])
    return config


def reg(config, dataloader):
    inputs = dataloader.__iter__()
    '''to nn'''
    with torch.no_grad():
        feats, scores_overlap, scores_saliency = config.model(inputs)

    pcd = inputs['points'][0]
    len_src = inputs['stack_lengths'][0][0]
    c_rot, c_trans = inputs['rot'], inputs['trans']
    correspondence = inputs['correspondences']

    src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
    src_raw = copy.deepcopy(src_pcd)
    tgt_raw = copy.deepcopy(tgt_pcd)
    src_feats, tgt_feats = feats[:len_src].detach().cpu(), feats[len_src:].detach().cpu()
    src_overlap, src_saliency = scores_overlap[:len_src].detach().cpu(), scores_saliency[:len_src].detach().cpu()
    tgt_overlap, tgt_saliency = scores_overlap[len_src:].detach().cpu(), scores_saliency[len_src:].detach().cpu()

    ########################################
    # do probabilistic sampling guided by the score
    src_scores = src_overlap * src_saliency
    tgt_scores = tgt_overlap * tgt_saliency

    ########################################
    # run ransac and draw registration
    tsfm = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=False)
    draw_registration_result(src_raw, tgt_raw, src_overlap, tgt_overlap, src_saliency, tgt_saliency, tsfm)


def main():
    """"""
    '''get frame and target'''
    pc_src_path = new_frame()
    pc_tgt_path = './data/face_from_mr.pcd'

    '''vis the data'''
    pc_ct = o3.io.read_point_cloud(pc_src_path)
    pc_cam = o3.io.read_point_cloud(pc_tgt_path)
    o3.visualization.draw_geometries([pc_ct, pc_cam])

    '''build dataset pc'''
    config = get_config()
    dataset = ThreeDMatchDemo(config, src_path=pc_src_path, tgt_path=pc_tgt_path)

    _, neighborhood_limits = get_dataloader(dataset=dataset,
                                            batch_size=config.batch_size,
                                            shuffle=True,
                                            num_workers=config.num_workers,
                                            )
    dataloader, _ = get_dataloader(dataset=dataset,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    num_workers=1,
                                    neighborhood_limits=neighborhood_limits)
    '''reg face in pc'''
    reg(config, dataloader)
    return


if __name__ == '__main__':
    main()
