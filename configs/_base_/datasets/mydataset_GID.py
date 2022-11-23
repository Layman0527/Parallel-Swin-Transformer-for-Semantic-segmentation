# dataset settings
dataset_type = 'MYDataset_GID'
data_root = 'data/my_dataset_GID'
img_norm_cfg = dict(
    mean=[87.38801289,95.85703825,90.7189491], std=[57.50289248,57.68473096,59.05680913], to_rgb=True)
crop_size = (256,256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(256,256), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256,256),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir',
        ann_dir='ann_dir',
        pipeline=train_pipeline,
        split="splits/train.txt"),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir',
        ann_dir='ann_dir',
        split="splits/val.txt",
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir',
        ann_dir='ann_dir',
        split="splits/test.txt",
        pipeline=test_pipeline))

