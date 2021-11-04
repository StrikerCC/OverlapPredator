import open3d as o3


def main():
    src_pc = o3.io.read_point_cloud("./data/human_models/head_models/model_women/3D_model.pcd")
    tgt_pc = o3.io.read_point_cloud('./data/human_data/000010.pcd')

    print('src', src_pc, src_pc.get_max_bound())
    print('tgt', tgt_pc, tgt_pc.get_max_bound())
    src_pc.paint_uniform_color((0, 0.5, 0.5))
    tgt_pc.paint_uniform_color((1, 0, 0))
    o3.visualization.draw_geometries([src_pc, tgt_pc])


if __name__ == '__main__':
    main()
