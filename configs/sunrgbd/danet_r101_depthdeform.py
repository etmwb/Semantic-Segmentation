#model settings
model_type='segmentation'
model = dict(
    type='DANet',
    pretrained='/home/zhouzuoyu/zzyai/Semantic-Segmentation/pretrained/ade20k/encnet_resnet101_ade.pth',
    backbone=dict(
            type='ResNet',
            depth=101,
            num_stages=4,
            strides=(1, 2, 1, 1),
            dilations=(1, 1, 2, 4),
            out_indices=(0, 1, 2, 3),
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            dcn=dict(dcn_type='depthdeform'),
            stage_with_dcn=(False, True, True, True),
            style='pytorch',
            norm_eval=False,
            deep_base=True),
    backbone_depth=dict(type='PointNet',
                        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    head=dict(
        type='DANetHead',
        in_channels=2048,
        out_channels=37,
        loss_cfg=dict(type='CrossEntropyLoss'),
        norm_cfg=dict(type='SyncBN', requires_grad=True))
)
dataset_type = 'SunrgbdDataset'
data_root = 'data/sunrgbd/'
mean_cfg = dict(img=[123.675, 116.28, 103.53], HHA=[134.109, 80.936, 91.963])
std_cfg = dict(img=[1., 1., 1.], HHA=[1., 1., 1.])
train_pipeline = [
    dict(type='LoadImageFromFile', label_minus=True),
    dict(type='Resize', scale=(0.5, 2.0), shorter_side=350),
    dict(type='PadCrop',
         pad_value=dict(img=[123.675, 116.28, 103.53], label=255, depth=0, HHA=[134.109, 80.936, 91.963]),
         crop_size=[500, 500]),
    dict(type='RandomFlip'),
    dict(type='RandomHSV', h_scale=[0.9, 1.1], s_scale=[0.9, 1.1], v_scale=[25, 25]),
    dict(type='Normalize', mean=mean_cfg, std=std_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=('HHA', ))
]
val_pipeline = [
    dict(type='LoadImageFromFile', label_minus=True),
    dict(type='Normalize', mean=mean_cfg, std=std_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=('HHA', ))
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=1,
    extra=dict(ignore_index=255,
               cls_weight=None),
    train=dict(type=dataset_type,
               path_file='train.txt',
               data_root=data_root,
               pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        path_file='test.txt',
        data_root=data_root,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        path_file='test.txt',
        data_root=data_root,
        pipeline=val_pipeline
    ))
evaluation=dict(ignore_index=255, interval=5)
# optimizer
# lr is set for a batch size of 4
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001,
                 paramgroup_options=[dict(params='backbone', lr_mult=1),
                                     dict(params='head', lr_mult=10),
                                     dict(params='backbone_depth', lr_mult=10)])
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, by_epoch=False)
checkpoint_config = dict(interval=-1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
#runtime settings
total_epochs = 180
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/sunrgbd/danet_r101_depthdeform'
load_from = None
resume_from = None
workflow = [('train', 1)]