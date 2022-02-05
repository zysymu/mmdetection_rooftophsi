# dataset settings
dataset_type = 'RhsiDataset'
data_root = 'data/RIT-HS20/'
train_pipeline = [
    dict(type='LoadMaskedHSIImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1600, 200), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadMaskedHSIImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 200),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
		type=dataset_type,
		ann_file=data_root + 'annotationsjson/instances_train_s10.json',
		img_prefix=data_root + 'spectral/',
		mask_prefix=data_root + 'masks/',
		pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotationsjson/instances_val.json',
        img_prefix=data_root + 'spectral/',
	mask_prefix=data_root + 'masks/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotationsjson/instances_val.json',
        img_prefix=data_root + 'spectral/',
	mask_prefix=data_root + 'masks/',
        pipeline=test_pipeline))
evaluation = dict(interval=2, metric='bbox', classwise=True)
