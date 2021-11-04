architectures = dict()

architectures['human'] = [
    'simple',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'nearest_upsample',
    'unary',
    'unary',
    'nearest_upsample',
    'unary',
    'last_unary'
]

architectures['indoor'] = [
    'simple',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'nearest_upsample',
    'unary',
    'nearest_upsample',
    'unary',
    'nearest_upsample',
    'last_unary'
]

architectures['kitti'] = [
    'simple',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'nearest_upsample',
    'unary',
    'nearest_upsample',
    'unary',
    'nearest_upsample',
    'last_unary'
]

architectures['modelnet'] = [
    'simple',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'nearest_upsample',
    'unary',
    'unary',
    'nearest_upsample',
    'unary',
    'last_unary'
]

"""
KPFCNN(
  (encoder_blocks): ModuleList(
    (0): SimpleBlock(
      (KPConv): KPConv(radius: 0.16, extent: 0.12, in_feat: 1, out_feat: 256)
      (batch_norm): BatchNormBlock(in_feat: 256, momentum: 0.020, only_bias: False)
      (leaky_relu): LeakyReLU(negative_slope=0.1)
    )
    (1): ResnetBottleneckBlock(
      (unary1): UnaryBlock(in_feat: 256, out_feat: 128, BN: True, ReLU: True)
      (KPConv): KPConv(radius: 0.16, extent: 0.12, in_feat: 128, out_feat: 128)
      (batch_norm_conv): BatchNormBlock(in_feat: 128, momentum: 0.020, only_bias: False)
      (unary2): UnaryBlock(in_feat: 128, out_feat: 512, BN: True, ReLU: False)
      (unary_shortcut): UnaryBlock(in_feat: 256, out_feat: 512, BN: True, ReLU: False)
      (leaky_relu): LeakyReLU(negative_slope=0.1)
    )
    (2): ResnetBottleneckBlock(
      (unary1): UnaryBlock(in_feat: 512, out_feat: 128, BN: True, ReLU: True)
      (KPConv): KPConv(radius: 0.16, extent: 0.12, in_feat: 128, out_feat: 128)
      (batch_norm_conv): BatchNormBlock(in_feat: 128, momentum: 0.020, only_bias: False)
      (unary2): UnaryBlock(in_feat: 128, out_feat: 512, BN: True, ReLU: False)
      (unary_shortcut): Identity()
      (leaky_relu): LeakyReLU(negative_slope=0.1)
    )
    (3): ResnetBottleneckBlock(
      (unary1): UnaryBlock(in_feat: 512, out_feat: 128, BN: True, ReLU: True)
      (KPConv): KPConv(radius: 0.16, extent: 0.12, in_feat: 128, out_feat: 128)
      (batch_norm_conv): BatchNormBlock(in_feat: 128, momentum: 0.020, only_bias: False)
      (unary2): UnaryBlock(in_feat: 128, out_feat: 512, BN: True, ReLU: False)
      (unary_shortcut): Identity()
      (leaky_relu): LeakyReLU(negative_slope=0.1)
    )
    (4): ResnetBottleneckBlock(
      (unary1): UnaryBlock(in_feat: 512, out_feat: 256, BN: True, ReLU: True)
      (KPConv): KPConv(radius: 0.33, extent: 0.24, in_feat: 256, out_feat: 256)
      (batch_norm_conv): BatchNormBlock(in_feat: 256, momentum: 0.020, only_bias: False)
      (unary2): UnaryBlock(in_feat: 256, out_feat: 1024, BN: True, ReLU: False)
      (unary_shortcut): UnaryBlock(in_feat: 512, out_feat: 1024, BN: True, ReLU: False)
      (leaky_relu): LeakyReLU(negative_slope=0.1)
    )
    (5): ResnetBottleneckBlock(
      (unary1): UnaryBlock(in_feat: 1024, out_feat: 256, BN: True, ReLU: True)
      (KPConv): KPConv(radius: 0.33, extent: 0.24, in_feat: 256, out_feat: 256)
      (batch_norm_conv): BatchNormBlock(in_feat: 256, momentum: 0.020, only_bias: False)
      (unary2): UnaryBlock(in_feat: 256, out_feat: 1024, BN: True, ReLU: False)
      (unary_shortcut): Identity()
      (leaky_relu): LeakyReLU(negative_slope=0.1)
    )
    (6): ResnetBottleneckBlock(
      (unary1): UnaryBlock(in_feat: 1024, out_feat: 256, BN: True, ReLU: True)
      (KPConv): KPConv(radius: 0.33, extent: 0.24, in_feat: 256, out_feat: 256)
      (batch_norm_conv): BatchNormBlock(in_feat: 256, momentum: 0.020, only_bias: False)
      (unary2): UnaryBlock(in_feat: 256, out_feat: 1024, BN: True, ReLU: False)
      (unary_shortcut): Identity()
      (leaky_relu): LeakyReLU(negative_slope=0.1)
    )
    (7): ResnetBottleneckBlock(
      (unary1): UnaryBlock(in_feat: 1024, out_feat: 512, BN: True, ReLU: True)
      (KPConv): KPConv(radius: 0.66, extent: 0.48, in_feat: 512, out_feat: 512)
      (batch_norm_conv): BatchNormBlock(in_feat: 512, momentum: 0.020, only_bias: False)
      (unary2): UnaryBlock(in_feat: 512, out_feat: 2048, BN: True, ReLU: False)
      (unary_shortcut): UnaryBlock(in_feat: 1024, out_feat: 2048, BN: True, ReLU: False)
      (leaky_relu): LeakyReLU(negative_slope=0.1)
    )
    (8): ResnetBottleneckBlock(
      (unary1): UnaryBlock(in_feat: 2048, out_feat: 512, BN: True, ReLU: True)
      (KPConv): KPConv(radius: 0.66, extent: 0.48, in_feat: 512, out_feat: 512)
      (batch_norm_conv): BatchNormBlock(in_feat: 512, momentum: 0.020, only_bias: False)
      (unary2): UnaryBlock(in_feat: 512, out_feat: 2048, BN: True, ReLU: False)
      (unary_shortcut): Identity()
      (leaky_relu): LeakyReLU(negative_slope=0.1)
    )
  )
  (bottle): Conv1d(2048, 256, kernel_size=(1,), stride=(1,))
  (gnn): GCN(
    (layers): ModuleList(
      (0): SelfAttention(
        (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (in1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (conv2): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (in2): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (conv3): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (in3): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
      (1): AttentionalPropagation(
        (attn): MultiHeadedAttention(
          (merge): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          (proj): ModuleList(
            (0): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (1): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
            (2): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
          )
        )
        (mlp): Sequential(
          (0): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
          (1): InstanceNorm1d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
          (2): ReLU()
          (3): Conv1d(512, 256, kernel_size=(1,), stride=(1,))
        )
      )
      (2): SelfAttention(
        (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (in1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (conv2): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (in2): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (conv3): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (in3): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
      )
    )
  )
  (proj_gnn): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
  (proj_score): Conv1d(256, 1, kernel_size=(1,), stride=(1,))
  (decoder_blocks): ModuleList(
    (0): NearestUpsampleBlock(layer: 2 -> 1)
    (1): UnaryBlock(in_feat: 1282, out_feat: 129, BN: True, ReLU: True)
    (2): UnaryBlock(in_feat: 129, out_feat: 129, BN: True, ReLU: True)
    (3): NearestUpsampleBlock(layer: 1 -> 0)
    (4): UnaryBlock(in_feat: 641, out_feat: 64, BN: True, ReLU: True)
    (5): LastUnaryBlock(in_feat: 64, out_feat: 98)
  )
)
"""