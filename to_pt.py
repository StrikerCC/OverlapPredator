# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 11/4/21 6:54 PM
"""
import torch
from configs.models import architectures
from models.architectures import KPFCNN
from lib.utils import load_config
from easydict import EasyDict as edict


def main():
    config = edict(load_config('./configs/train/head.yaml'))
    config.architecture = architectures['human']
    model = KPFCNN(config)
    checkpoint = torch.load('/home/cheng/proj/3d/3DPcReg/weights/model_best_recall.pth')
    model.load_state_dict(checkpoint['state_dict'])

    sm = torch.jit.script(model)
    sm.save('/home/cheng/proj/3d/3DPcReg/weights/model_best_recall_sm.pt')


if __name__ == '__main__':
    main()
