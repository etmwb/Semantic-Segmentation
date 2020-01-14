#model settings
model_type='segmentation'
model = dict(
    type='DANet',
    pretrained='/home/zhouzuoyu/zzyai/Semantic-Segmentation/pretrained/ade20k/fcn_resnet50_ade.pth',
    backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            strides=(1, 2, 1, 1),
            dilations=(1, 1, 2, 4),
            out_indices=(0, 1, 2, 3),
            dcn=dict(dcn_type='depthaware'),
            stage_with_dcn=(False, True, True, True),
            style='pytorch',
            norm_eval=False),
    head=dict(
        type='DANetHead',
        in_channels=2048,
        out_channels=40,
        loss_cfg=dict(type='CrossEntropyLoss'))
)
dataset_type = 'Nyuv2Dataset'
data_root = 'data/nyuv2/'
mean_cfg = dict(img=[123.675, 116.28, 103.53], HHA=[132.431, 94.076, 118.477])
std_cfg = dict(img=[1., 1., 1.], HHA=[1., 1., 1.])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(0.5, 2.0), shorter_side=350),
    dict(type='PadCrop',
         pad_value=dict(img=[123.675, 116.28, 103.53], label=255, depth=0, HHA=[132.431, 94.076, 118.477]),
         crop_size=[500, 500]),
    dict(type='RandomFlip'),
    # dict(type='RandomHSV', h_scale=[0.9, 1.1], s_scale=[0.9, 1.1], v_scale=[25, 25]),
    dict(type='Normalize', mean=mean_cfg, std=std_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=('depth', ))
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Normalize', mean=mean_cfg, std=std_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=('depth', ))
]
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=2,
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
evaluation=dict(ignore_index=255)
# optimizer
# lr is set for a batch size of 4
optimizer = dict(type='SGD', lr=0.00025, momentum=0.9, weight_decay=0.0001,
                 paramgroup_options=[dict(params='backbone', lr_mult=1),
                                     dict(params='head', lr_mult=10)])
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
total_epochs = 100
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/nyuv2/danet_r50_depthdeform'
load_from = None
resume_from = None
workflow = [('train', 1)]