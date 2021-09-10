# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 9/3/21 10:00 AM
"""
import copy

from data_faker.multiview_merge import combine_pcs
from data_faker.dataset import Writer, Reader
from data_faker.vis import draw_registration_result


def main():
    flag_merge_multiview = False
    flag_write = False

    if flag_merge_multiview:
        model_dir_path = 'data_raw/human_models/head_models/model_women/'  # output file name
        combine_pcs(model_dir_path)

    sample_path = 'data_raw/'
    output_path = 'data/human_data_tiny/'
    output_json_path = output_path + 'data.json'

    if flag_write:
        ds = Writer()
        ds.write(sample_path, output_path, output_json_path)

    dl = Reader()
    dl.read(output_json_path)

    i = -10
    data = dl[i]
    pc_model = data['pc_model']
    pc_artificial = data['pc_artificial']
    tf = data['pose']
    pc_model_ = copy.deepcopy(pc_model)
    pc_model_.transform(tf)
    draw_registration_result(source=pc_artificial)
    draw_registration_result(source=pc_artificial, target=pc_model_)
    print(dl[i])


if __name__ == '__main__':
    main()
